# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for tt/model_config.py (no device).

Run device-hidden (the root conftest opens the UMD cluster even for collection), e.g. with the wrapper::

    scripts/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib -q \
        models/demos/motif3/tests/unit/test_infra_config.py
"""

import ast
import dataclasses
import importlib
import inspect
import json
import math
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.motif3.tt import generator_api as api
from models.demos.motif3.tt import prefill_plan as pp
from models.demos.motif3.tt.model_config import (
    CACHE_FORMAT_VERSION,
    COMPUTE_ROLES,
    DEFAULT_HF_META_DIR,
    DEFAULT_L1_SMALL_SIZE,
    DEFAULT_WEIGHTS_DIR,
    DEFAULT_WIDE_STEP_RATIO,
    EXPERTS_DOWN_GRID_WIDE,
    FP32_ACC_OFF_ROLES,
    ROUTER_EXACT_FP32_DECODE_ROWS,
    SP1_GLOBAL_CHUNKS,
    LayerSpec,
    MeshAxes,
    MotifTTConfig,
    compute_config_descriptor,
    device_params,
    experts_down_pc,
    experts_gate_up_pc,
    flash_mla_decode_pc,
    lm_head_pc,
    make_compute_kernel_config,
    mcast1d_matmul_pc,
    mesh_l1_small_bytes,
    mesh_shape_from_env,
    require_l1_small,
    resolve_weights_dir,
    resumed_prefill_pc,
    rope_scaling_of,
    router_decode_pc,
    sdpa_prefill_chunks,
    sdpa_prefill_pc,
    sp1_global_chunks,
)
from models.demos.motif3.tt.rope import inv_freq_for_kind

HF_META = str(DEFAULT_HF_META_DIR)
_ENV = (
    "MOTIF3_NUM_LAYERS",
    "MOTIF3_KV_POOL_TOKENS",
    "MOTIF3_MAX_MODEL_LEN",
    "MOTIF3_TRACE_REGION_SIZE",
    "MOTIF3_FABRIC",
    "MOTIF3_L1_SMALL_SIZE",
    "MOTIF3_ROUTER_LOGITS",
    "MESH_DEVICE",
    "MOTIF3_WEIGHTS_DIR",
    "HF_MODEL",
    "TT_CACHE_PATH",
    "MOTIF3_TT_CACHE_PATH",
    "TT_MODEL_WEIGHTS_REVISION",
    # features (docs/features/FEATURES_DESIGN.md §1.3)
    "MOTIF3_PREFIX_CACHING",
    "MOTIF3_CHUNKED_PREFILL",
    "MOTIF3_SPEC_DECODE",
    "MOTIF3_KV_REPLICATED_DECODE",
    "MOTIF3_PREFILL_MAX_BUCKET",
    "MOTIF3_PACKED_PREFILL",
    "MOTIF3_SPEC_VERIFY",
    "MOTIF3_RING_GATHER",
    # P5 / T64 (docs/p5_t64/P5_T64_DESIGN.md §3.9, §4.7)
    "MOTIF3_PACKED_PREFILL_MAX_SEG",
    "MOTIF3_PACKED_PREFILL_MAX_TOKENS",
    "MOTIF3_PACKED_PREFILL_PK1",
    "MOTIF3_PACKED_WARMUP",
    "MOTIF3_WIDE_MIN_LANES",
    # C1: TT weight-cache policy and the T64 / T32 step ratio
    "MOTIF3_TT_CACHE_POLICY",
    "MOTIF3_WIDE_STEP_RATIO",
    # Phase A (docs/OPTIMIZATION_PLAN.md §4.3): chunk budget, FlashMLA SWA cores, router mask
    "MOTIF3_CHUNK_BUDGET",
    "MOTIF3_FLASH_MLA_SWA_MCPH",
    "MOTIF3_ROUTER_MASK",
    # Phase B (docs/OPT_PHASE_A_REVIEW.md §7.2): B1 sparse decode experts
    "MOTIF3_DECODE_EXPERTS",
    # B3 fused decode MoE PolyNorm
    "MOTIF3_MOE_POLYNORM",
    # B5 fused decode shared-expert PolyNorm
    "MOTIF3_SHARED_POLYNORM",
    # B6a host input staging / replay wait
    "MOTIF3_HOST_STAGING",
    "MOTIF3_HOST_WAIT",
    # B6b asynchronous decode (the bridge's supports_async_decode)
    "MOTIF3_ASYNC_DECODE",
    # B2a token-compacted prefill MoE
    "MOTIF3_PREFILL_MOE",
    "MOTIF3_PREFILL_MOE_BLOCK",
    "MOTIF3_PREFILL_MOE_MIN_ROWS",
    # the capture thread (E3)
    "MOTIF3_CAPTURE_THREAD",
    # B7 traced prefill
    "MOTIF3_PREFILL_TRACE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)


def _cfg(**kw):
    return MotifTTConfig.from_hf_config(HF_META, **kw)


def _fields(cfg):
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


def _weights_dir_or_skip():
    for d in (DEFAULT_WEIGHTS_DIR, DEFAULT_HF_META_DIR):
        if (d / "config.json").is_file() and (d / "configuration_motif.py").is_file():
            return d
    pytest.skip("no Motif-3 config + remote code dir")


@pytest.mark.parametrize("mesh_shape, tp_axis", [((4, 8), 1), ((8, 4), 0)])
def test_axes_and_partition(mesh_shape, tp_axis):
    cfg = _cfg(mesh_shape=mesh_shape)
    a = cfg.axes
    assert a.tp_axis == tp_axis and a.dp_axis == 1 - tp_axis
    assert (cfg.tp, cfg.dp, cfg.num_chips) == (8, 4, 32)
    assert (cfg.q_heads_per_chip, cfg.kv_groups_per_chip, cfg.signal_heads_per_chip) == (10, 2, 8)
    assert cfg.experts_per_chip == 12 and cfg.lanes_per_row == 8
    assert cfg.dense_intermediate_per_chip == 1536 and cfg.shared_intermediate_per_chip == 160
    assert cfg.vocab_per_chip == 27520 and cfg.latent_proj_dim == 1664 and cfg.kv_latent_dim == 576

    # coord <-> roles are inverse, chip index is dp * 8 + tp
    for dp in range(cfg.dp):
        for tp in range(cfg.tp):
            r, c = a.coord(dp, tp)
            assert a.roles(r, c) == (dp, tp)
            assert a.chip_index(dp, tp) == 8 * dp + tp

    # heads: each chip owns 2 whole groups; every q / signal head exactly once, contiguous
    q, s, g = [], [], []
    for tp in range(cfg.tp):
        h = cfg.chip_heads(tp)
        assert list(h.groups) == [2 * tp, 2 * tp + 1]
        assert list(h.q_heads) == list(range(10 * tp, 10 * tp + 10))
        assert list(h.signal_heads) == list(range(8 * tp, 8 * tp + 8))
        # signal s = 4g + j of the local groups; q head 5g + j
        assert sorted(4 * gg + j for gg in h.groups for j in range(4)) == list(h.signal_heads)
        assert sorted(5 * gg + j for gg in h.groups for j in range(5)) == list(h.q_heads)
        q += list(h.q_heads)
        s += list(h.signal_heads)
        g += list(h.groups)
    assert q == list(range(80)) and s == list(range(64)) and g == list(range(16))

    # experts: chip k = 8 dp + tp holds [12k, 12k+12); chip_of_expert is the inverse
    seen = []
    for dp in range(cfg.dp):
        for tp in range(cfg.tp):
            ex = cfg.experts_of_chip(dp, tp)
            k = 8 * dp + tp
            assert list(ex) == list(range(12 * k, 12 * k + 12))
            assert all(cfg.chip_of_expert(e) == (dp, tp) for e in ex)
            seen += list(ex)
    assert seen == list(range(384))

    # lanes: lane l on DP row l // 8
    assert [cfg.lane_row(l) for l in range(32)] == [l // 8 for l in range(32)]
    assert list(cfg.row_lanes(2)) == list(range(16, 24))

    # role-based mesh dims
    assert a.mesh_dims(dp_dim=0, tp_dim=None) == ((0, None) if tp_axis == 1 else (None, 0))
    assert a.mesh_dims(dp_dim=0, tp_dim=1) == ((0, 1) if tp_axis == 1 else (1, 0))
    assert a.tag == f"mesh{mesh_shape[0]}x{mesh_shape[1]}"


def test_small_meshes():
    assert MeshAxes.detect((1, 8)).tp_axis == 1
    assert MeshAxes.detect((1, 1)).tp_size == 1
    cfg = _cfg(mesh_shape=(1, 8))
    assert cfg.dp == 1 and cfg.lanes_per_row == 32 and cfg.experts_per_chip == 48


def test_layer_schedule():
    cfg = _cfg()
    assert cfg.global_layers == tuple(range(0, 53, 4)) and len(cfg.global_layers) == 14
    assert len(cfg.swa_layers) == 39
    assert cfg.dense_layers == (0, 1) and cfg.moe_layers == tuple(range(2, 53))
    for L in cfg.layers:
        if L.is_global:
            assert L.window is None and L.rope_kind == "yarn" and L.attn_kind == "global"
            assert abs(L.softmax_scale - 0.14467963) < 1e-8
        else:
            assert L.window == 129 and L.sliding_window_size == 129 and L.rope_kind == "plain"
            assert L.attn_kind == "swa"
            assert abs(L.softmax_scale - 0.07216878) < 1e-8
    assert abs(cfg.softmax_scale(0) - 192**-0.5 * (0.1 * math.log(64) + 1) ** 2) < 1e-12
    assert cfg.eos_token_ids == (0, 3, 6) and cfg.bos_token_id == 1
    assert (cfg.rms_norm_eps, cfg.mhc_rms_eps, cfg.polynorm_eps) == (1e-5, 1e-6, 1e-6)
    assert (cfg.yarn_factor, cfg.yarn_original_max_pos, cfg.yarn_beta_fast, cfg.yarn_beta_slow) == (
        64.0,
        4096,
        32.0,
        1.0,
    )
    assert cfg.rope_type == "yarn" and cfg.yarn_theta == 1e4
    assert cfg.mhc_h_post_coeff == 1.0 and cfg.sinkhorn_iters == 20 and cfg.n_streams == 4


def test_layer_schedule_matches_reference():
    """Cross-check with the CPU golden package (imported lazily: its API may still change)."""
    ref_cfg = pytest.importorskip("models.demos.motif3.reference.config")
    args = ref_cfg.MotifArgs.from_hf_config(HF_META)
    cfg = _cfg()
    for i in range(cfg.num_layers):
        L = cfg.layer(i)
        assert L.is_swa == args.is_swa_layer(i)
        assert L.is_moe == args.is_moe_layer(i)
        assert L.window == args.attention_window(i)
        assert L.softmax_scale == pytest.approx(args.softmax_scale(i), rel=1e-15)
        assert (L.rope_kind == "yarn") == args.uses_yarn(i)


# ======================================================================================================
# INFRA-1: YaRN survives every way of handing over the HF config
# ======================================================================================================
def test_from_hf_config_object_dict_and_path_agree():
    """``AutoConfig(trust_remote_code)`` object (what vLLM's ``hf_config`` is), its transformers-5 ``to_dict()``
    (``rope_parameters``, no ``rope_scaling``) and the ``config.json`` path give identical configs: every field,
    the layer schedule and the YaRN ``inv_freq``. Before INFRA-1 the object path lost YaRN (factor 1.0, plain RoPE
    on the 14 global layers, inv_freq off by 5.4e-3) and the EOS set (0,) instead of (0, 3, 6)."""
    transformers = pytest.importorskip("transformers")
    wdir = _weights_dir_or_skip()
    by_path = MotifTTConfig.from_hf_config(str(wdir))
    obj = transformers.AutoConfig.from_pretrained(str(wdir), trust_remote_code=True)
    d = obj.to_dict()
    assert "rope_parameters" in d  # transformers 5.x: the YaRN dict moved here ...
    by_obj = MotifTTConfig.from_hf_config(obj)
    by_dict = MotifTTConfig.from_hf_config(d)  # ... and to_dict() carries _name_or_path -> generation_config.json
    yarn = inv_freq_for_kind(by_path, "yarn")
    for name, other in (("object", by_obj), ("to_dict", by_dict)):
        diff = {k: (v, _fields(other)[k]) for k, v in _fields(by_path).items() if _fields(other)[k] != v}
        assert not diff, f"{name}: fields differ from the config.json path: {diff}"
        assert other.layers == by_path.layers, name
        assert torch.equal(inv_freq_for_kind(other, "yarn"), yarn), name
        assert torch.equal(inv_freq_for_kind(other, "plain"), inv_freq_for_kind(by_path, "plain")), name
    assert by_obj.rope_type == "yarn" and by_obj.yarn_factor == 64.0 and by_obj.yarn_original_max_pos == 4096
    assert all(by_obj.layer(i).rope_kind == "yarn" for i in by_obj.global_layers)
    assert by_obj.eos_token_ids == (0, 3, 6)


def test_rope_parameters_only_dicts():
    """Synthetic transformers-5 dicts: flat ``rope_parameters`` and the per-layer-type form both keep YaRN; an
    explicitly absent rope dict falls back to plain RoPE (no silent YaRN invention)."""
    raw = json.load(open(f"{HF_META}/config.json"))
    ref = MotifTTConfig.from_hf_config(raw)
    rs = dict(raw["rope_scaling"])
    flat = {k: v for k, v in raw.items() if k != "rope_scaling"}
    flat["rope_parameters"] = {**rs, "attention_factor": 1.0}  # what to_dict() adds
    nested = {k: v for k, v in raw.items() if k != "rope_scaling"}
    nested["rope_parameters"] = {"full_attention": rs, "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}}
    no_theta = dict(flat)
    no_theta.pop("rope_theta")  # some transformers versions move rope_theta into rope_parameters only
    for name, d in (("flat", flat), ("nested", nested), ("no_top_level_theta", no_theta)):
        cfg = MotifTTConfig.from_hf_config(d)
        assert _fields(cfg) == _fields(ref), name
        assert torch.equal(inv_freq_for_kind(cfg, "yarn"), inv_freq_for_kind(ref, "yarn")), name
    assert rope_scaling_of(nested) == rs and rope_scaling_of({"rope_scaling": None}) == {}
    plain = MotifTTConfig.from_hf_config({k: v for k, v in raw.items() if k != "rope_scaling"})
    assert plain.rope_type == "default" and plain.yarn_factor == 1.0
    assert all(L.rope_kind == "plain" for L in plain.layers)


def test_eos_from_generation_config_next_to_name_or_path(tmp_path):
    raw = json.load(open(f"{HF_META}/config.json"))
    assert MotifTTConfig.from_hf_config(raw).eos_token_ids == (0,)  # a bare dict has no generation_config
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [0, 3, 6], "bos_token_id": 1}))
    assert MotifTTConfig.from_hf_config({**raw, "_name_or_path": str(tmp_path)}).eos_token_ids == (0, 3, 6)

    class FakeObj:  # duck-typed PretrainedConfig (vLLM hands over a dynamic class; never isinstance'd)
        _name_or_path = str(tmp_path)

        def to_dict(self):
            return dict(raw)

    assert MotifTTConfig.from_hf_config(FakeObj()).eos_token_ids == (0, 3, 6)


# ======================================================================================================
# INFRA-2 / INFRA-7: KV pool geometry (bridge formula, N from allocate_kv_cache), buckets
# ======================================================================================================
def test_kv_pool_and_buckets():
    cfg = _cfg()
    assert cfg.kv_block_size == 64 and cfg.kv_pool_tokens == 262144 and cfg.max_batch == 32
    assert cfg.max_num_seqs == 32 and cfg.kv_num_blocks_actual is None
    # the bridge's get_max_tokens_all_users (+32 null-block reserve) through the plugin's formula
    assert cfg.kv_num_blocks == cfg.kv_num_blocks_expected == math.ceil((262144 + 32 + 64 * 32) / 64) == 4129
    assert cfg.kv_num_blocks == api.expected_num_blocks() == api.plugin_num_blocks(262144 + 32, 64, 32)
    assert cfg.kv_pool_tokens_allocated == 4129 * 64
    assert cfg.kv_blocks_per_seq == 512 and cfg.kv_cache_shape == (4129, 1, 64, 576)
    assert cfg.kv_cache_bytes_per_chip() == 53 * 4129 * 2 * 18 * 1088 == 8_571_407_616  # design §1.1: 8.57 GB
    assert cfg.kv_cache_bytes_per_chip() == api.kv_cache_bytes_per_chip(4129, 64, 53, "bfp8")
    assert cfg.prefill_buckets == (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    assert cfg.prefill_buckets == api.prefill_buckets(32768)
    assert [cfg.prefill_bucket(n) for n in (1, 128, 129, 4000, 32768)] == [128, 128, 256, 4096, 32768]
    with pytest.raises(ValueError):
        cfg.prefill_bucket(32769)
    assert [cfg.prefill_page_table_entries(b) for b in (128, 4096, 32768)] == [2, 64, 512]
    with pytest.raises(ValueError):
        cfg.prefill_page_table_entries(65536)


def test_kv_geometry_from_allocate_kv_cache():
    """N comes from allocate_kv_cache (M2/M3): --max-num-seqs 8 makes the plugin allocate 4105 blocks; the hint wins
    once recorded; block 32 is supported, 128 / 16 are not (BRIDGE-1)."""
    cfg = _cfg(max_num_seqs=8)
    assert cfg.max_batch == 32 and cfg.lanes_per_row == 8  # the trace always runs 32 lanes (M3)
    assert cfg.kv_num_blocks == 4105 == api.expected_num_blocks(262144, 64, 8)
    cfg.set_kv_geometry(4129, 64)
    assert cfg.kv_num_blocks == 4129 and cfg.kv_cache_shape == (4129, 1, 64, 576) and cfg.kv_num_blocks_expected == 4105
    cfg.set_kv_geometry(8225, 32)
    assert cfg.kv_cache_shape == (8225, 1, 32, 576) and cfg.kv_blocks_per_seq == 1024
    for bad in ((4129, 128), (4129, 16), (1, 64)):
        with pytest.raises(ValueError):
            cfg.set_kv_geometry(*bad)
    assert cfg.kv_cache_shape == (8225, 1, 32, 576)  # a refused geometry leaves the config unchanged
    with pytest.raises(ValueError):
        _cfg(max_num_seqs=33)


def test_max_model_len_alignment_and_cdiv_width():
    """INFRA-7: no `% block` hard error; W = cdiv(max_model_len, block) like the bridge; max_model_len must still be
    tile aligned. Serving additionally requires a multiple of 256 (BRIDGE-3, enforced by generator_api)."""
    cfg = _cfg(max_model_len=4128)  # 4128 = 64.5 blocks of 64: allowed, W rounds up
    assert cfg.kv_blocks_per_seq == 65 and cfg.prefill_buckets[-1] == 4128
    with pytest.raises(ValueError, match="multiple of 32"):
        _cfg(max_model_len=4100)
    with pytest.raises(ValueError, match="multiple of 256"):
        api.check_max_model_len(4128)
    assert api.check_max_model_len(4096) == 4096


def test_dtypes_compute_and_cache_paths(tmp_path, monkeypatch):
    cfg = _cfg()
    d = cfg.dtypes
    assert d.routed_experts == ttnn.bfloat8_b and d.attention == ttnn.bfloat16 and d.kv_cache == ttnn.bfloat8_b
    assert d.router == ttnn.bfloat16 and d.mhc == ttnn.bfloat16 and d.lm_head == ttnn.bfloat16
    assert d.embedding == ttnn.bfloat16 and d.router_bias == ttnn.float32 and d.activations == ttnn.bfloat16
    assert d.tag == "e8s8d8a16r16m16l16v16" and d.kv_cache_name == "bfp8"
    ck = cfg.compute_config("router")
    assert ck.math_fidelity == ttnn.MathFidelity.HiFi4 and ck.fp32_dest_acc_en and not ck.math_approx_mode
    ce = cfg.compute_config("experts")
    assert ce.math_fidelity == ttnn.MathFidelity.HiFi4 and ce.fp32_dest_acc_en  # G6: HiFi4 is free
    assert cfg.compute_config("router") is ck  # cached
    with pytest.raises(KeyError):
        cfg.compute_config("nope")
    with pytest.raises(KeyError, match="sdpa_decode"):
        cfg.compute_config("sdpa")  # split by INFRA-3
    bf16kv = _cfg(kv_cache_dtype="bf16")
    assert bf16kv.dtypes.kv_cache == ttnn.bfloat16 and bf16kv.dtypes.tag == d.tag  # KV dtype is not cached
    assert bf16kv.kv_cache_bytes_per_chip() == api.kv_cache_bytes_per_chip(4129, 64, 53, "bf16")

    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("MOTIF3_NUM_LAYERS", "3")
    monkeypatch.setenv("MOTIF3_KV_POOL_TOKENS", "131072")
    c2 = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(8, 4))
    assert c2.num_layers == 3 and len(c2.layers) == 3 and c2.num_hidden_layers == 53
    assert c2.kv_num_blocks == (131072 + 32) // 64 + 32 + 1 == api.expected_num_blocks(131072, 64, 32)
    tag = f"motif3-2ed2ed5c-c{CACHE_FORMAT_VERSION}-e8s8d8a16r16m16l16v16"
    assert c2.cache_version_tag == tag
    assert c2.cache_dir == tmp_path / tag / "mesh8x4"
    assert c2.cache_file("attn.wq_b", 2) == tmp_path / tag / "mesh8x4" / "L02" / "attn.wq_b"
    assert c2.cache_file("lm_head", None) == tmp_path / tag / "mesh8x4" / "global" / "lm_head"
    monkeypatch.setenv("MOTIF3_KV_POOL_TOKENS", "1000")  # same validation as the bridge
    with pytest.raises(ValueError, match="MOTIF3_KV_POOL_TOKENS"):
        MotifTTConfig.from_hf_config(HF_META)


# ======================================================================================================
# INFRA-3 / INFRA-4: compute roles and program configs are the ones the gates validated
# ======================================================================================================
def _ckc_fields(c):
    return (c.math_fidelity, c.math_approx_mode, c.fp32_dest_acc_en, c.packer_l1_acc, c.dst_full_sync_en)


class _FakeMesh:
    """Just enough mesh for the gates' config helpers (they only ask for the compute grid)."""

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(12, 10)


def test_compute_roles_match_gates():
    """Every role is HiFi4 with math approx off and packer L1 accumulation off (INFRA-3); fp32 acc everywhere except
    SDPA prefill (G2 ran it off). Configs are compared with the gate files' own helpers, so a drift on either side
    fails here. G6 (experts) and G8 (rope) ran packer_l1_acc on; the infra device test measured G6 both ways (identical
    PCC / latency), so those two are compared without that field."""
    g1 = importlib.import_module("models.demos.motif3.tests.unit.gates.test_g1_mla_decode")
    g2 = importlib.import_module("models.demos.motif3.tests.unit.gates.test_g2_sdpa_prefill")
    g6 = importlib.import_module("models.demos.motif3.tests.unit.gates.test_g6_experts")
    g8 = importlib.import_module("models.demos.motif3.tests.unit.gates.test_g8_rope")
    gu = importlib.import_module("models.demos.motif3.tests.unit.gates.gate_utils")
    cfg = _cfg()
    assert set(COMPUTE_ROLES) >= {
        "attn_latent",
        "attn_heads",
        "sdpa_decode",
        "sdpa_prefill",
        "rope",
        "norm",
        "mhc",
        "router",
        "polynorm",
        "experts",
        "shared",
        "dense_mlp",
        "lm_head",
        "eltwise",
        "sdpa_prefill_fp32",
        "ccl_reduce",
    }
    assert "sdpa" not in COMPUTE_ROLES
    assert FP32_ACC_OFF_ROLES == {"sdpa_prefill"}
    sp, spf = cfg.compute_role("sdpa_prefill"), cfg.compute_role("sdpa_prefill_fp32")
    assert (spf.fidelity, spf.approx, spf.packer_l1_acc) == (sp.fidelity, sp.approx, sp.packer_l1_acc) and spf.fp32_acc
    for role, r in COMPUTE_ROLES.items():
        c = cfg.compute_config(role)
        assert c.math_fidelity == ttnn.MathFidelity.HiFi4 and not c.math_approx_mode, role
        assert c.fp32_dest_acc_en == (role != "sdpa_prefill"), role
        assert not c.packer_l1_acc and not r.packer_l1_acc, role
        assert r.source, role
    gate = {
        "sdpa_decode": g1.mla_compute_cfg(fp32_acc=True),  # G1 recommended: HiFi4, fp32 acc, packer off
        "sdpa_prefill": gu.compute_cfg(**{k: v for k, v in g2.CKC["hifi4"].items()}),
        "experts": g6.mm_cfg("HiFi4"),
        "rope": g8.ckc_of("hifi4_fp32acc"),
        "router": gu.compute_cfg("HiFi4", fp32_acc=True, approx=False, packer_l1_acc=False),  # G5 DeviceRouter
        "polynorm": gu.hifi4(fp32_acc=True),  # G6 polynorm_tt rms_norm
        "mhc": gu.hifi4(fp32_acc=True),  # G3
    }
    packer_on_in_gate = {"experts", "rope"}
    for role, want in gate.items():
        got, exp = _ckc_fields(cfg.compute_config(role)), _ckc_fields(want)
        if role in packer_on_in_gate:
            assert exp[3] and not got[3], role  # gate ran packer_l1_acc on; the role keeps INFRA-3's default off
            got, exp = got[:3] + got[4:], exp[:3] + exp[4:]
        assert got == exp, role
    assert not make_compute_kernel_config("HiFi4", True).packer_l1_acc  # default off (INFRA-3)


def test_program_config_builders_match_gates():
    g1 = importlib.import_module("models.demos.motif3.tests.unit.gates.test_g1_mla_decode")
    g2 = importlib.import_module("models.demos.motif3.tests.unit.gates.test_g2_sdpa_prefill")
    g6 = importlib.import_module("models.demos.motif3.tests.unit.gates.test_g6_experts")
    mesh = _FakeMesh()
    cfg = _cfg()
    # G1: explicit k_chunk 128 (never 0 / None), q_chunk 0, exp approx off, 16 cores per head batch
    for pc in (flash_mla_decode_pc(mesh), flash_mla_decode_pc((12, 10)), cfg.flash_mla_decode_pc()):
        assert repr(pc) == repr(g1.program_config(mesh, 128))
        assert pc.k_chunk_size == 128 and pc.q_chunk_size == 0 and pc.max_cores_per_head_batch == 16
    # A2: SWA layers 4 cores per head batch (bitwise equal to 16 on the 129-key window), global layers keep 16
    for pc in (flash_mla_decode_pc(mesh, "swa"), cfg.flash_mla_decode_pc("swa"), cfg.flash_mla_decode_pc(1)):
        assert pc.max_cores_per_head_batch == 4 and pc.k_chunk_size == 128 and pc.q_chunk_size == 0
    assert repr(cfg.flash_mla_decode_pc(0)) == repr(cfg.flash_mla_decode_pc("global")) == repr(g1.program_config(mesh, 128))
    # G2: 128/128 on SWA, 256/256 on global, exp approx off (the HiFi4 configuration)
    assert repr(sdpa_prefill_pc("swa", mesh)) == repr(g2.sdpa_pc(mesh, 128, 128, exp_approx=False))
    assert repr(sdpa_prefill_pc("global", mesh)) == repr(g2.sdpa_pc(mesh, 256, 256, exp_approx=False))
    assert repr(cfg.sdpa_prefill_pc(0)) == repr(sdpa_prefill_pc("global", mesh))  # layer 0 is global
    assert repr(cfg.sdpa_prefill_pc(cfg.layer(1))) == repr(sdpa_prefill_pc("swa", mesh))
    assert sdpa_prefill_chunks("global", 128) == (128, 128)  # G2 ran S = 128 with 128/128 on both kinds
    assert sdpa_prefill_chunks("global", 32768) == (256, 256) and sdpa_prefill_chunks("swa", 4096) == (128, 128)
    with pytest.raises(ValueError):
        sdpa_prefill_chunks("global", 384)
    with pytest.raises(ValueError):
        sdpa_prefill_pc("ring", mesh)
    # G6: 1D multicast, gate_up on 10 x 8 cores (per_core_N 1), down on 8 x 4 cores (per_core_N 4)
    assert repr(experts_gate_up_pc()) == repr(g6.mcast1d_cfg((10, 8), 80, 8)) == repr(cfg.experts_gate_up_pc())
    assert repr(experts_down_pc()) == repr(g6.mcast1d_cfg((8, 4), 128, 4)) == repr(cfg.experts_down_pc())
    gu_pc, dn_pc = experts_gate_up_pc(), experts_down_pc()
    assert (gu_pc.in0_block_w, gu_pc.per_core_M, gu_pc.per_core_N, gu_pc.out_subblock_w) == (8, 1, 1, 1)
    assert (gu_pc.fuse_batch, gu_pc.mcast_in0) == (False, True)
    assert (dn_pc.in0_block_w, dn_pc.per_core_N, dn_pc.out_subblock_w) == (4, 4, 4)


# ======================================================================================================
# misc
# ======================================================================================================
def test_device_params(monkeypatch):
    """Attention P0: every mesh opens with an L1_SMALL region of 32768 B (the CCL semaphores live there)."""
    p = device_params()
    assert p["fabric_config"] == ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    assert p["trace_region_size"] == 268435456
    assert p["l1_small_size"] == DEFAULT_L1_SMALL_SIZE == api.L1_SMALL_SIZE == 32768
    assert _cfg().device_params()["l1_small_size"] == 32768 and _cfg().l1_small_size == 32768
    attn = importlib.import_module("models.demos.motif3.tt.attention")
    assert attn.RECOMMENDED_L1_SMALL_SIZE == DEFAULT_L1_SMALL_SIZE  # the module's requirement == the shared default
    monkeypatch.setenv("MOTIF3_FABRIC", "FABRIC_1D_RING")
    assert device_params()["fabric_config"] == ttnn.FabricConfig.FABRIC_1D_RING
    assert device_params(trace_region_size=1 << 20, l1_small_size=16384) == {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "trace_region_size": 1 << 20,
        "l1_small_size": 16384,
    }
    assert device_params(l1_small_size=0)["l1_small_size"] == 0  # only to reproduce the hazard
    monkeypatch.setenv("MOTIF3_L1_SMALL_SIZE", "65536")
    assert device_params()["l1_small_size"] == 65536 and _cfg().l1_small_size == 65536
    monkeypatch.delenv("MOTIF3_L1_SMALL_SIZE")
    with pytest.raises(ValueError):
        device_params(l1_small_size=-1)
    with pytest.raises(ValueError):
        device_params("FABRIC_NOPE")
    # host objects never count as meshes for the L1_SMALL queries (no Metal context is created)
    assert mesh_l1_small_bytes(SimpleNamespace(shape=(4, 8))) is None
    assert require_l1_small(SimpleNamespace(shape=(4, 8))) is None
    monkeypatch.setenv("MESH_DEVICE", "(8, 4)")
    assert mesh_shape_from_env() == (8, 4)
    monkeypatch.setenv("MESH_DEVICE", "4,8")
    assert mesh_shape_from_env() == (4, 8)
    monkeypatch.delenv("MESH_DEVICE")
    assert mesh_shape_from_env() == (4, 8)
    # a host-side fake mesh never queries the Metal context for the fabric (INFRA-6 applies to real meshes only)
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=SimpleNamespace(shape=(4, 8)))
    assert cfg.fabric == "FABRIC_1D_RING" and cfg.mesh_shape == (4, 8)


def test_from_dict_and_validation():
    d = json.load(open(f"{HF_META}/config.json"))
    cfg = MotifTTConfig.from_hf_config(d, mesh_shape=(4, 8))
    assert cfg.n_heads == 80 and cfg.eos_token_ids == (0,)  # no generation_config next to a dict
    with pytest.raises(ValueError):
        MotifTTConfig.from_hf_config(d, mesh_shape=(4, 8), num_experts=100)
    with pytest.raises(ValueError):
        MotifTTConfig.from_hf_config(d, mesh_shape=(4, 8), num_layers=60)


def test_from_settings(tmp_path):
    """GEN-1 mapping: config from <weights>/config.json, max_batch stays 32 whatever max_num_seqs is, KV dtype,
    block size, cache root, revision; an uncached weights location falls back to the hf_config object."""
    wdir = _weights_dir_or_skip()
    s = api.GeneratorSettings(
        max_batch_size=8,
        max_seq_len=4096,
        num_layers=3,
        kv_cache_dtype="bf16",
        weights_path=str(wdir),
        weights_revision="2ed2ed5cfabffa10fdabb2fc0d0288f8e6de893a",
        cache_path=str(tmp_path),
        block_size=32,
    )
    cfg = MotifTTConfig.from_settings(s, mesh_shape=(4, 8))
    assert (cfg.num_layers, cfg.max_model_len, cfg.max_batch, cfg.max_num_seqs, cfg.lanes_per_row) == (
        3,
        4096,
        32,
        8,
        8,
    )
    assert cfg.dtypes.kv_cache == ttnn.bfloat16 and cfg.kv_block_size == 32 and cfg.weights_dir == wdir
    assert cfg.tt_cache_root == tmp_path and cfg.eos_token_ids == (0, 3, 6) and cfg.rope_type == "yarn"
    assert cfg.kv_num_blocks == api.expected_num_blocks(262144, 32, 8) and cfg.kv_blocks_per_seq == 128
    raw = json.load(open(wdir / "config.json"))
    s2 = dataclasses.replace(s, weights_path="Motif-Technologies/Motif-3", block_size=None, weights_source="test")
    cfg2 = MotifTTConfig.from_settings(s2, mesh_shape=(8, 4), hf_config=raw, num_layers=2)
    assert cfg2.num_layers == 2 and cfg2.kv_block_size == 64 and cfg2.axes.tp_axis == 0
    assert cfg2.weights_dir == resolve_weights_dir()  # not local: the generator resolves / downloads it itself


def test_resolve_weights_dir_order(tmp_path, monkeypatch):
    """One precedence order with the bridge (BRIDGE-2): MOTIF3_WEIGHTS_DIR > HF_MODEL dir > HF-cache snapshot of a
    repo-id HF_MODEL at TT_MODEL_WEIGHTS_REVISION > the local default."""
    assert resolve_weights_dir() == DEFAULT_WEIGHTS_DIR
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("HF_MODEL", str(b))
    assert resolve_weights_dir() == b
    monkeypatch.setenv("MOTIF3_WEIGHTS_DIR", str(a))
    assert resolve_weights_dir() == a
    monkeypatch.delenv("MOTIF3_WEIGHTS_DIR")
    hub = tmp_path / "hub"
    snap = hub / "models--Org--Motif-3" / "snapshots" / ("ab" * 20)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_MODEL", "Org/Motif-3")
    monkeypatch.setenv("TT_MODEL_WEIGHTS_REVISION", "ab" * 20)
    assert resolve_weights_dir() == snap
    monkeypatch.setenv("TT_MODEL_WEIGHTS_REVISION", "cd" * 20)  # not cached: host tools fall back to the default
    assert resolve_weights_dir() == DEFAULT_WEIGHTS_DIR
    monkeypatch.setenv("MOTIF3_WEIGHTS_DIR", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="MOTIF3_WEIGHTS_DIR"):
        resolve_weights_dir()


# ======================================================================================================
# wave-B1 shared changes: module program configs, generic_op descriptors, PolyNorm fields, module defaults
# ======================================================================================================
def _local(module: str, name: str):
    """A module's local helper, or None once the module switched to the shared builder (then nothing to compare)."""
    mod = importlib.import_module(f"models.demos.motif3.tt.{module}")
    return getattr(mod, name, None)


def test_module_program_configs_match_the_modules():
    """Every shared builder requested in wave B1 returns exactly the module's measured local config (repr equal),
    so a module can switch to ``cfg.*_pc()`` without a behaviour change (README §5)."""
    cfg = _cfg()
    compared = []
    f = _local("attention", "decode_matmul_program_configs")
    if f is not None:
        loc, sh = f(cfg), cfg.attn_decode_matmul_pcs()
        assert set(loc) == set(sh)
        for k in loc:
            assert repr(loc[k]) == repr(sh[k]) == repr(cfg.attn_decode_matmul_pc(k)), k
        compared.append("attention")
    f = _local("mlp", "build_decode_program_configs")
    if f is not None:
        for kind in ("dense", "shared"):
            loc, lf = f(kind, cfg.mlp_decode_dims(kind), cfg.compute_grid)
            sh, sf = cfg.mlp_decode_pcs(kind)
            assert lf == sf and set(loc) == set(sh), kind
            for k in loc:
                assert repr(loc[k]) == repr(sh[k]) == repr(cfg.mlp_decode_pc(kind, k)), (kind, k)
        assert _local("mlp", "DECODE_MATMUL_GRIDS") == importlib.import_module(
            "models.demos.motif3.tt.model_config"
        ).MLP_DECODE_MATMUL_GRIDS
        compared.append("mlp")
    mc = importlib.import_module("models.demos.motif3.tt.model_config")
    f = _local("mlp", "decode_matmul_pc")
    if f is not None:
        for k, n, g, bw in ((4096, 3072, (12, 8), 8), (1536, 4096, (8, 4), 8), (160, 4096, (8, 4), 5), (4096, 320, (10, 1), 32)):
            assert repr(f(k, n, g, in0_block_w=bw)) == repr(mc.decode_matmul_pc(k, n, g, in0_block_w=bw))
        with pytest.raises(ValueError):
            mc.decode_matmul_pc(4096, 320, (12, 8))  # 10 tiles cannot keep 96 cores busy
    moe = importlib.import_module("models.demos.motif3.tt.moe")
    router = getattr(getattr(moe, "MotifRouter", None), "_decode_pc", None)
    if router is not None:
        stub = SimpleNamespace(n_experts=cfg.num_experts, cfg=cfg)
        for sig in (False, True):
            assert repr(router(stub, sigmoid=sig)) == repr(cfg.router_decode_pc(sigmoid=sig)), sig
        assert cfg.router_decode_pc(sigmoid=True).fused_activation is not None
        compared.append("moe.router")
    f = _local("moe", "prefill_experts_pc")
    if f is not None:
        for m in (128, 1024, 2048, 4096, 8192):
            assert repr(f(m // 32, 80, out_block_w=20)) == repr(cfg.experts_prefill_gate_up_pc(m)), m
            assert repr(f(m // 32, 128, out_block_w=16)) == repr(cfg.experts_prefill_down_pc(m)), m
        assert cfg.experts_prefill_gate_up_pc(1024) is None and cfg.experts_prefill_down_pc(4096) is not None
        compared.append("moe.experts_prefill")
    lm = importlib.import_module("models.demos.motif3.tt.lm_head")
    f = getattr(lm, "lm_head_program_config", None)
    if f is not None:
        for split in ("mesh", "tp"):
            assert repr(f(cfg, split)) == repr(cfg.lm_head_pc(split)), split
            assert cfg.vocab_per_shard(split) == lm.vocab_per_shard(cfg, split)
        assert cfg.lm_head_pc("mesh", "auto") is None
        compared.append("lm_head")
    f = getattr(lm, "sharded_norm_configs", None)
    if f is not None:
        m1, p1 = f(cfg)
        m2, p2 = cfg.decode_norm_configs()
        assert repr(m1) == repr(m2) and repr(p1) == repr(p2)
        compared.append("decode_norm")
    f = _local("mhc", "_bmm_pc")
    if f is not None:
        assert repr(f(cfg.compute_grid, 16)) == repr(cfg.mhc_decode_proj_pc(32))
        compared.append("mhc")
    print(f"[config] shared builders == module helpers for {compared}")


def test_mcast1d_partial_last_block_and_fused_activation():
    """embed_head: ``per_core_n`` allows a partial last block (215 vocab tiles on 108 cores); the G6 even split and
    its errors are unchanged; moe: the router's fused sigmoid is a builder argument."""
    pc = mcast1d_matmul_pc((12, 9), 215, 16, per_core_n=2, fuse_batch=True)
    assert (pc.per_core_N, pc.out_block_w, pc.out_subblock_w, pc.fuse_batch) == (2, 2, 2, True)
    assert mcast1d_matmul_pc((12, 9), 860, 16, per_core_n=8, fuse_batch=True).out_subblock_w == 4
    with pytest.raises(ValueError):
        mcast1d_matmul_pc((12, 9), 220, 16, per_core_n=2)  # 110 blocks > 108 cores
    with pytest.raises(ValueError):
        mcast1d_matmul_pc((12, 9), 215, 16)  # no per_core_n: N must split evenly (G6)
    sig = mcast1d_matmul_pc((12, 1), 12, 32, 128, fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID))
    assert sig.fused_activation is not None and mcast1d_matmul_pc((12, 1), 12, 32, 128).fused_activation is None
    assert mcast1d_matmul_pc((2, 1), 6, 1, per_core_n=3, fp32_acc=False).out_subblock_w == 3


def test_compute_config_descriptor_for_generic_op_kernels():
    """sinkhorn kernel: a generic_op takes a ComputeConfigDescriptor (per-CB unpack-to-dest modes); the shared helper
    builds it from a role and equals the kernels' own descriptors."""
    d = compute_config_descriptor("mhc", fp32_unpack_cbs=(0, 5, 63))
    r = COMPUTE_ROLES["mhc"]
    assert d.math_fidelity == getattr(ttnn.MathFidelity, r.fidelity) and d.fp32_dest_acc_en == r.fp32_acc
    assert d.math_approx_mode == r.approx and not d.dst_full_sync_en
    modes = list(d.unpack_to_dest_mode)
    assert len(modes) == 64 and [i for i, m in enumerate(modes) if m == ttnn.UnpackToDestMode.UnpackToDestFp32] == [0, 5, 63]
    assert compute_config_descriptor("router", dst_full_sync_en=True).dst_full_sync_en
    with pytest.raises(ValueError):
        compute_config_descriptor("mhc", fp32_unpack_cbs=(64,))
    with pytest.raises(KeyError, match="sdpa_decode"):
        compute_config_descriptor("sdpa")
    sm = importlib.import_module("models.demos.motif3.tt.kernels.sinkhorn_motif")
    if hasattr(sm, "_compute_config"):
        mine = compute_config_descriptor("mhc", fp32_unpack_cbs=(sm.CB_MIXES, sm.CB_CONSTS, sm.CB_TMP))
        theirs = sm._compute_config()
        for f in ("math_fidelity", "fp32_dest_acc_en", "math_approx_mode", "dst_full_sync_en"):
            assert getattr(mine, f) == getattr(theirs, f), f
        assert list(mine.unpack_to_dest_mode) == list(theirs.unpack_to_dest_mode)


def test_polynorm_semantics_fields():
    """mlp review issue 7: polynorm_sigmoid_weight (default True; False is rejected) and the per-layer output scales
    are parsed from the config, and polynorm_output_scale_for_layer follows the reference."""
    cfg = _cfg()
    assert cfg.polynorm_sigmoid_weight is True and cfg.polynorm_output_scale_per_layer == {}
    assert cfg.polynorm_output_scale_for_layer(7) == 0.5
    d = json.load(open(f"{HF_META}/config.json"))
    d["polynorm_output_scale_per_layer"] = {"3": 0.25, "40": 1.0}
    c2 = MotifTTConfig.from_hf_config(d, mesh_shape=(4, 8))
    assert c2.polynorm_output_scale_per_layer == {3: 0.25, 40: 1.0}
    assert (c2.polynorm_output_scale_for_layer(3), c2.polynorm_output_scale_for_layer(4)) == (0.25, 0.5)
    pn = importlib.import_module("models.demos.motif3.tt.polynorm")
    assert pn.polynorm_output_scale(c2, 3) == 0.25 and pn.polynorm_output_scale(c2, 4) == 0.5
    ref = importlib.import_module("models.demos.motif3.reference.config").MotifArgs.from_hf_config(d)
    assert ref.polynorm_output_scale_for_layer(3) == c2.polynorm_output_scale_for_layer(3)
    d["polynorm_sigmoid_weight"] = False
    with pytest.raises(NotImplementedError, match="polynorm_sigmoid_weight"):
        MotifTTConfig.from_hf_config(d, mesh_shape=(4, 8))
    with pytest.raises(ValueError):
        MotifTTConfig.from_hf_config(HF_META, polynorm_output_scale_per_layer={60: 0.5})


def test_flash_mla_swa_mcph_knob(monkeypatch):
    """A2 (docs/OPTIMIZATION_PLAN.md §3.3; logs/opt/phaseA/A2): ``MOTIF3_FLASH_MLA_SWA_MCPH`` sets the FlashMLA decode
    ``max_cores_per_head_batch`` of the SWA layers (default 4; 16 = the release config); the global layers keep 16 at
    every setting, and a value outside [2, 16] (1 included: the bitwise argument needs >= 2 k-chunk cores; review I-5)
    is refused when the config is built."""
    import models.demos.motif3.tt.model_config as mc

    assert mc.FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH == 16 and mc.FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH_SWA == 4
    cfg = _cfg()
    assert cfg.flash_mla_swa_mcph == 4
    swa = [l for l in range(cfg.num_layers) if cfg.layer(l).attn_kind == "swa"]
    glo = [l for l in range(cfg.num_layers) if cfg.layer(l).attn_kind == "global"]
    assert len(swa) == 39 and len(glo) == 14
    assert {cfg.flash_mla_decode_pc(cfg.layer(l)).max_cores_per_head_batch for l in swa} == {4}
    assert {cfg.flash_mla_decode_pc(l).max_cores_per_head_batch for l in glo} == {16}
    assert "mla_mcph swa=4/global=16" in cfg.describe()
    for v, want in (("16", 16), ("2", 2), ("", 4)):
        monkeypatch.setenv("MOTIF3_FLASH_MLA_SWA_MCPH", v)
        c = _cfg()
        assert c.flash_mla_swa_mcph == want and c.flash_mla_decode_pc("swa").max_cores_per_head_batch == want
        assert c.flash_mla_decode_pc("global").max_cores_per_head_batch == 16
    assert mc.FLASH_MLA_DECODE_MIN_CORES_PER_HEAD_BATCH_SWA == 2
    for bad in ("0", "1", "17"):
        monkeypatch.setenv("MOTIF3_FLASH_MLA_SWA_MCPH", bad)
        with pytest.raises(ValueError, match="MOTIF3_FLASH_MLA_SWA_MCPH"):
            _cfg()
    monkeypatch.delenv("MOTIF3_FLASH_MLA_SWA_MCPH")
    assert _cfg(flash_mla_swa_mcph=8).flash_mla_decode_pc("swa").max_cores_per_head_batch == 8
    with pytest.raises(ValueError):
        mc.flash_mla_decode_pc((12, 10), "swa", swa_mcph=32)
    for bad in (1, 2.5):
        with pytest.raises(ValueError):
            mc.flash_mla_decode_pc((12, 10), "swa", swa_mcph=bad)
    with pytest.raises(ValueError):
        _cfg(flash_mla_swa_mcph=1)


def test_router_mask_knob(monkeypatch):
    """A5 (docs/OPTIMIZATION_PLAN.md §3.3; logs/opt/phaseA/A5): ``MOTIF3_ROUTER_MASK`` selects the decode routing-weight
    path, ``gather`` (default: the release), ``scatter`` (``MotifRouter.route_local``) or ``fused`` (B4,
    ``MotifRouter.route_fused`` + ``kernels/router_topk``); neither is bitwise equal to the release, so both stay off
    until the Validate gates; case and blanks ignored, anything else refused; ``describe`` shows it."""
    from models.demos.motif3.tt.model_config import ROUTER_MASK_MODES

    assert ROUTER_MASK_MODES == ("gather", "scatter", "fused")
    assert _cfg().router_mask == "gather" and "router_mask=gather" in _cfg().describe()
    for v, want in (("scatter", "scatter"), (" Scatter ", "scatter"), ("gather", "gather"), ("", "gather"),
                    ("fused", "fused"), (" FUSED ", "fused")):
        monkeypatch.setenv("MOTIF3_ROUTER_MASK", v)
        assert _cfg().router_mask == want, v
    monkeypatch.setenv("MOTIF3_ROUTER_MASK", "threshold")
    with pytest.raises(ValueError, match="MOTIF3_ROUTER_MASK"):
        _cfg()
    monkeypatch.delenv("MOTIF3_ROUTER_MASK")
    assert _cfg(router_mask="scatter").router_mask == "scatter"
    assert _cfg(router_mask="fused").router_mask == "fused" and "router_mask=fused" in _cfg(router_mask="fused").describe()
    with pytest.raises(ValueError, match="router_mask"):
        _cfg(router_mask="ge")


def test_decode_experts_knob(monkeypatch):
    """B1 (docs/OPTIMIZATION_PLAN.md §3.3; logs/opt/phaseA/M6): ``MOTIF3_DECODE_EXPERTS`` selects the decode routed
    experts, ``dense`` (default: the release) or ``sparse`` (``ttnn.sparse_matmul`` skips the local experts no live row
    routes to); case and blanks ignored, anything else refused; ``describe`` shows it."""
    from models.demos.motif3.tt.model_config import DECODE_EXPERTS_MODES

    assert DECODE_EXPERTS_MODES == ("dense", "sparse")
    assert _cfg().decode_experts == "dense" and "decode_experts=dense" in _cfg().describe()
    for v, want in (("sparse", "sparse"), (" Sparse ", "sparse"), ("dense", "dense"), ("", "dense")):
        monkeypatch.setenv("MOTIF3_DECODE_EXPERTS", v)
        assert _cfg().decode_experts == want, v
    monkeypatch.setenv("MOTIF3_DECODE_EXPERTS", "indices")
    with pytest.raises(ValueError, match="MOTIF3_DECODE_EXPERTS"):
        _cfg()
    monkeypatch.delenv("MOTIF3_DECODE_EXPERTS")
    assert _cfg(decode_experts="sparse").decode_experts == "sparse"
    with pytest.raises(ValueError, match="decode_experts"):
        _cfg(decode_experts="skip")


def test_moe_polynorm_knob(monkeypatch):
    """B3 (docs/OPTIMIZATION_PLAN.md §3.3; logs/opt/phaseA/M10): ``MOTIF3_MOE_POLYNORM`` selects the decode routed-expert
    PolyNorm, ``composite`` (default: the release) or ``fused`` (one generic_op, not bitwise equal: off until the shared
    eval); case and blanks ignored, anything else refused; ``describe`` shows it."""
    from models.demos.motif3.tt.model_config import MOE_POLYNORM_MODES

    assert MOE_POLYNORM_MODES == ("composite", "fused")
    assert _cfg().moe_polynorm == "composite" and "moe_polynorm=composite" in _cfg().describe()
    for v, want in (("fused", "fused"), (" Fused ", "fused"), ("composite", "composite"), ("", "composite")):
        monkeypatch.setenv("MOTIF3_MOE_POLYNORM", v)
        assert _cfg().moe_polynorm == want, v
    monkeypatch.setenv("MOTIF3_MOE_POLYNORM", "horner")
    with pytest.raises(ValueError, match="MOTIF3_MOE_POLYNORM"):
        _cfg()
    monkeypatch.delenv("MOTIF3_MOE_POLYNORM")
    assert _cfg(moe_polynorm="fused").moe_polynorm == "fused"
    with pytest.raises(ValueError, match="moe_polynorm"):
        _cfg(moe_polynorm="kernel")


def test_shared_polynorm_knob(monkeypatch):
    """B5 (docs/OPTIMIZATION_PLAN.md §3.3): ``MOTIF3_SHARED_POLYNORM`` selects the decode shared-expert PolyNorm,
    ``composite`` (default: the release) or ``fused`` (tt/kernels/shared_polynorm.py, bitwise equal by construction);
    case and blanks ignored, anything else refused; ``describe`` shows it."""
    from models.demos.motif3.tt.model_config import SHARED_POLYNORM_MODES

    assert SHARED_POLYNORM_MODES == ("composite", "fused")
    assert _cfg().shared_polynorm == "composite" and "shared_polynorm=composite" in _cfg().describe()
    for v, want in (("fused", "fused"), (" FUSED ", "fused"), ("composite", "composite"), ("", "composite")):
        monkeypatch.setenv("MOTIF3_SHARED_POLYNORM", v)
        assert _cfg().shared_polynorm == want, v
    monkeypatch.setenv("MOTIF3_SHARED_POLYNORM", "kernel")
    with pytest.raises(ValueError, match="MOTIF3_SHARED_POLYNORM"):
        _cfg()
    monkeypatch.delenv("MOTIF3_SHARED_POLYNORM")
    assert _cfg(shared_polynorm="fused").shared_polynorm == "fused"
    with pytest.raises(ValueError, match="shared_polynorm"):
        _cfg(shared_polynorm="horner")


def test_prefill_moe_knobs(monkeypatch):
    """B2a (docs/OPTIMIZATION_PLAN.md §3.3 B2; logs/opt/phaseB/B2a): ``MOTIF3_PREFILL_MOE`` selects the prefill routed
    experts, ``compact`` (default: token-compacted, bitwise equal) or ``dense`` (the release); ``MOTIF3_PREFILL_MOE_BLOCK``
    the block rows (``auto`` | 32 | 64 | 128) and ``MOTIF3_PREFILL_MOE_MIN_ROWS`` the smallest compacted chunk (a
    multiple of 32, default 1024). Case and blanks ignored, anything else refused; ``describe`` shows them."""
    from models.demos.motif3.tt.model_config import (DEFAULT_PREFILL_MOE_MIN_ROWS, PREFILL_MOE_BLOCKS,
                                                     PREFILL_MOE_MODES)

    assert PREFILL_MOE_MODES == ("dense", "compact") and PREFILL_MOE_BLOCKS == ("auto", "32", "64", "128")
    c = _cfg()
    assert (c.prefill_moe, c.prefill_moe_block, c.prefill_moe_min_rows) == ("compact", "auto",
                                                                              DEFAULT_PREFILL_MOE_MIN_ROWS)
    assert DEFAULT_PREFILL_MOE_MIN_ROWS == 1024 and "prefill_moe=compact/auto/1024" in c.describe()
    for v, want in (("compact", "compact"), (" Dense ", "dense"), ("dense", "dense"), ("", "compact")):
        monkeypatch.setenv("MOTIF3_PREFILL_MOE", v)
        assert _cfg().prefill_moe == want, v
    monkeypatch.setenv("MOTIF3_PREFILL_MOE", "sparse")
    with pytest.raises(ValueError, match="MOTIF3_PREFILL_MOE"):
        _cfg()
    monkeypatch.delenv("MOTIF3_PREFILL_MOE")
    for v, want in (("64", "64"), (" AUTO ", "auto"), ("", "auto"), ("128", "128")):
        monkeypatch.setenv("MOTIF3_PREFILL_MOE_BLOCK", v)
        assert _cfg().prefill_moe_block == want, v
    monkeypatch.setenv("MOTIF3_PREFILL_MOE_BLOCK", "48")
    with pytest.raises(ValueError, match="MOTIF3_PREFILL_MOE_BLOCK"):
        _cfg()
    monkeypatch.delenv("MOTIF3_PREFILL_MOE_BLOCK")
    monkeypatch.setenv("MOTIF3_PREFILL_MOE_MIN_ROWS", "512")
    assert _cfg().prefill_moe_min_rows == 512
    for bad in ("500", "0"):
        monkeypatch.setenv("MOTIF3_PREFILL_MOE_MIN_ROWS", bad)
        with pytest.raises(ValueError, match="MOTIF3_PREFILL_MOE_MIN_ROWS"):
            _cfg()
    monkeypatch.delenv("MOTIF3_PREFILL_MOE_MIN_ROWS")
    assert _cfg(prefill_moe="dense").prefill_moe == "dense"
    with pytest.raises(ValueError, match="prefill_moe"):
        _cfg(prefill_moe="fast")


def test_host_staging_and_wait_knobs(monkeypatch):
    """B6a (docs/OPT_PHASE_A_REVIEW.md §7.1; logs/opt/phaseB/B6a): ``MOTIF3_HOST_STAGING`` (``fast`` default |
    ``release``) and ``MOTIF3_HOST_WAIT`` (``spin`` default | ``block``) are host-only decode knobs (bitwise neutral:
    the defaults since the B6a gates); case and blanks ignored, anything else refused; ``describe`` shows both."""
    from models.demos.motif3.tt.model_config import HOST_STAGING_MODES, HOST_WAIT_MODES

    assert HOST_STAGING_MODES == ("release", "fast") and HOST_WAIT_MODES == ("block", "spin")
    c = _cfg()
    assert c.host_staging == "fast" and c.host_wait == "spin"
    assert "host_staging=fast host_wait=spin" in c.describe()
    for v, want in (("fast", "fast"), (" RELEASE ", "release"), ("release", "release"), ("", "fast")):
        monkeypatch.setenv("MOTIF3_HOST_STAGING", v)
        assert _cfg().host_staging == want, v
    for v, want in (("spin", "spin"), (" Block", "block"), ("block", "block"), ("", "spin")):
        monkeypatch.setenv("MOTIF3_HOST_WAIT", v)
        assert _cfg().host_wait == want, v
    monkeypatch.setenv("MOTIF3_HOST_STAGING", "turbo")
    with pytest.raises(ValueError, match="MOTIF3_HOST_STAGING"):
        _cfg()
    monkeypatch.delenv("MOTIF3_HOST_STAGING")
    monkeypatch.setenv("MOTIF3_HOST_WAIT", "busy")
    with pytest.raises(ValueError, match="MOTIF3_HOST_WAIT"):
        _cfg()
    monkeypatch.delenv("MOTIF3_HOST_WAIT")
    assert _cfg(host_staging="release", host_wait="block").host_wait == "block"
    with pytest.raises(ValueError, match="host_wait"):
        _cfg(host_wait="poll")


def test_module_defaults(monkeypatch):
    """Defaults the decoder passes: the Motif Sinkhorn kernel (Option B) and the composite router (decision D1 pending,
    MOTIF3_ROUTER_LOGITS=exact_fp32 for the model-level A/B)."""
    cfg = _cfg()
    assert cfg.mhc_sinkhorn == "motif" and cfg.router_logits == "composite"
    monkeypatch.setenv("MOTIF3_ROUTER_LOGITS", "exact_fp32")
    assert _cfg().router_logits == "exact_fp32"
    monkeypatch.setenv("MOTIF3_ROUTER_LOGITS", "fast")
    with pytest.raises(ValueError, match="router_logits"):
        _cfg()
    monkeypatch.delenv("MOTIF3_ROUTER_LOGITS")
    with pytest.raises(ValueError, match="mhc_sinkhorn"):
        _cfg(mhc_sinkhorn="direct")
    assert _cfg(mhc_sinkhorn="stock").mhc_sinkhorn == "stock"
    mhc = importlib.import_module("models.demos.motif3.tt.mhc")
    assert set(getattr(mhc, "SINKHORN_IMPLS", ("motif", "stock"))) == {"motif", "stock"}


def test_serving_tt_config_contract():
    """The vLLM "tt" additional config: l1_small_size is mandatory (the plugin opens the mesh with exactly that)."""
    tt = api.SERVING_TT_CONFIG
    assert tt["l1_small_size"] == api.L1_SMALL_SIZE and tt["trace_mode"] == "decode_only"
    assert tt["fabric_config"] == "FABRIC_2D_TORUS_XY" and tt["dispatch_core_axis"] == "col"
    assert tt["trace_region_size"] == _cfg().trace_region_size
    assert api.serving_additional_config(trace_region_size=1)["tt"]["trace_region_size"] == 1
    assert api.serving_additional_config()["tt"] == tt and api.serving_additional_config()["tt"] is not tt
    assert api.check_tt_config(tt) == tt
    for bad in (None, {}, {"trace_mode": "decode_only"}, {"l1_small_size": 16384}, {"l1_small_size": "x"}):
        with pytest.raises(ValueError, match="l1_small_size|L1_SMALL"):
            api.check_tt_config(bad)
    assert api.check_tt_config({"l1_small_size": 65536})["l1_small_size"] == 65536


def test_ccl_dispatch_predicates():
    """tt/ccl.py mirrors ttnn's CCL path choice on the host (which path keeps its semaphores where): the draft-1
    payloads land on the expected branches (measured on device by test_infra_l1_small.py)."""
    ccl = importlib.import_module("models.demos.motif3.tt.ccl")
    # all_reduce scatter dim: [1,1,8|32,4096] -> dim 3 (128 tiles); [1,1,8,32] fp32 moments -> none (AG + local sum)
    assert ccl.finding_scatter_dim([1, 1, 32, 4096], 4, True, 8) == 3
    assert ccl.finding_scatter_dim([1, 1, 32, 32], 4, True, 8) == 4
    assert ccl.finding_scatter_dim([1, 1, 8, 4096], 4, False, 8) == 3
    # direct reduce-scatter: ring axis, <= 512 KiB per chip, whole-tile slices
    elig = ccl.direct_rs_eligible
    assert elig([1, 1, 8, 4096], [1, 1, 32, 4096], "BFLOAT16", True, False, 3, 8, True)  # decode AR(tp) 64 KB
    assert elig([1, 1, 32, 4096], [1, 1, 32, 4096], "FLOAT32", True, False, 3, 8, True)  # 512 KiB: the gate is <=
    assert not elig([1, 1, 32, 4096], [1, 1, 32, 4096], "BFLOAT16", True, False, 3, 4, False)  # DP line (TORUS_Y)
    assert not elig([1, 1, 128, 4096], [1, 1, 128, 4096], "BFLOAT16", True, False, 3, 8, True)  # prefill: 1 MiB
    assert not elig([1, 1, 8, 4096], [1, 1, 8, 4096], "BFLOAT16", False, False, 3, 8, True)  # ROW_MAJOR
    assert not elig([1, 1, 8, 4096], [1, 1, 32, 4096], "BFLOAT16", True, False, 2, 8, True)  # padded scatter dim
    assert not elig([1, 1, 8, 4096], [1, 1, 32, 4096], "BFLOAT16", True, True, 3, 8, True)  # sharded: ring path
    # composite all_gather: padded TILE gather dim (8 lanes), unaligned ROW_MAJOR rows
    assert ccl.composite_ag([1, 1, 8, 4096], [1, 1, 32, 4096], True, "BFLOAT16", 2)
    assert not ccl.composite_ag([1, 1, 32, 4096], [1, 1, 32, 4096], True, "BFLOAT16", 2)
    assert not ccl.composite_ag([1, 1, 8, 4096], [1, 1, 8, 4096], False, "BFLOAT16", 2)  # ag_dp_rows (RM, dim 2)
    assert ccl.composite_ag([1, 1, 8, 8], [1, 1, 8, 8], False, "FLOAT32", 3)  # 32 B rows
    assert ccl.composite_ag_for_ar([1, 1, 8, 32], True, 2) and not ccl.composite_ag_for_ar([1, 1, 8, 4096], True, 3)
    assert ccl.composite_rs([1, 1, 8, 4096], False, 3, 8)  # ROW_MAJOR
    assert ccl.composite_rs([1, 1, 32, 128], True, 3, 8)  # 16-column slices
    assert not ccl.composite_rs([1, 1, 8, 4096], True, 3, 8)


class _FakeMeshShape:
    def __init__(self, shape):
        self.shape = shape


def test_ccl_size1_axes_return_new_tensors(monkeypatch):
    """moe request: on a size-1 axis every MotifCCL collective returns a new tensor (a clone), never its input, so a
    caller's unconditional free of the input is safe on (1, 8) test meshes too."""
    ccl_mod = importlib.import_module("models.demos.motif3.tt.ccl")
    clones = []

    def fake_clone(x, memory_config=None):
        clones.append((x, memory_config))
        return SimpleNamespace(src=x)

    monkeypatch.setattr(ttnn, "clone", fake_clone)
    mesh = _FakeMeshShape((1, 8))
    ccl = ccl_mod.MotifCCL(mesh, MotifTTConfig.from_hf_config(HF_META, mesh_shape=(1, 8)))
    assert ccl.axis_size("dp") == 1 and not ccl.l1_small_semaphores  # a host fake has no L1_SMALL region
    x = SimpleNamespace(layout=ttnn.TILE_LAYOUT, shape=[1, 1, 8, 4096])
    for name, call in (
        ("all_reduce", lambda: ccl.ar_dp(x)),
        ("all_gather", lambda: ccl.ag_dp(x, 2)),
        ("reduce_scatter", lambda: ccl.rs_dp(x, 2)),
        ("partition", lambda: ccl.partition(x, 2, "dp")),
        ("ar_exact", lambda: ccl.ar_exact(x, "dp")),
        ("ag_dp_rows", lambda: ccl.ag_dp_rows(x)),
    ):
        out = call()
        assert out is not x and out.src is x, name
    assert all(mc == ttnn.DRAM_MEMORY_CONFIG for _, mc in clones) and len(clones) == 6


def test_tt_cache_root_precedence(tmp_path, monkeypatch):
    """CONV-2 / TIS runbook open issue 4: MOTIF3_TT_CACHE_PATH > TT_CACHE_PATH > motif-3/tt_cache, the same order for
    the bridge's GeneratorSettings and MotifTTConfig (so a converted cache can be shared without a symlink)."""
    from models.demos.motif3.tt.model_config import DEFAULT_TT_CACHE_ROOT, resolve_tt_cache_root

    assert resolve_tt_cache_root() == DEFAULT_TT_CACHE_ROOT and api.resolve_tt_cache_path({}) is None
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path / "tis"))
    assert resolve_tt_cache_root() == tmp_path / "tis" and _cfg().tt_cache_root == tmp_path / "tis"
    monkeypatch.setenv("MOTIF3_TT_CACHE_PATH", str(tmp_path / "motif"))
    assert resolve_tt_cache_root() == tmp_path / "motif" and _cfg().tt_cache_root == tmp_path / "motif"
    assert api.resolve_tt_cache_path({"TT_CACHE_PATH": "/a", "MOTIF3_TT_CACHE_PATH": " "}) == "/a"
    hf = SimpleNamespace(num_hidden_layers=53)
    env = {"TT_CACHE_PATH": "/tis/cache", "MOTIF3_TT_CACHE_PATH": "/shared/cache"}
    s = api.GeneratorSettings.from_env(hf, max_batch_size=32, max_seq_len=32768, environ=env)
    assert s.cache_path == "/shared/cache"
    s = api.GeneratorSettings.from_env(hf, max_batch_size=32, max_seq_len=32768, environ={"TT_CACHE_PATH": "/tis/cache"})
    assert s.cache_path == "/tis/cache"


# ======================================================================================================
# Features (docs/features/FEATURES_DESIGN.md §2, §3.1-§3.6; WP1): config fields and the generator contract
# ======================================================================================================
def test_features_config_defaults():
    """Span cap 8192 (D8), A = 128 (gate G9's per-bucket sp1 q / k), W' = 640, SWA tail 128, draft-1 KV writes, no
    MTP cache until spec is on."""
    cfg = _cfg()
    assert cfg.prefill_span_cap == api.DEFAULT_PREFILL_SPAN_CAP == 8192 and cfg.max_prefill_span == 8192
    assert cfg.prefill_span_buckets == (128, 256, 512, 1024, 2048, 4096, 8192) == pp.span_buckets(32768, 8192)
    assert cfg.prefill_buckets[-1] == 32768  # the draft-1 bucket list is unchanged
    assert cfg.prefill_swa_tail == 128 == pp.DEFAULT_SWA_TAIL
    assert cfg.prefill_resume_alignment == api.DEFAULT_PREFILL_ALIGNMENT == 128
    assert cfg.prefill_resume_alignment == pp.resume_alignment(64, cfg.prefill_span_buckets)
    assert cfg.sp1_page_table_width == 640 == pp.sdpa_table_width(32768, 8192, 64)
    assert (
        pp.recommended_budget(cfg.max_prefill_span, cfg.prefill_resume_alignment) == 8064
    )  # --max-num-batched-tokens = --long-prefill-token-threshold
    assert cfg.kv_write_mode == "row" and cfg.kv_replicated_decode is False and cfg.spec_tokens == 0
    assert cfg.mtp_kv_layers == 0 and cfg.kv_pool_layers == 53
    assert cfg.kv_pool_bytes_per_chip() == cfg.kv_cache_bytes_per_chip()
    assert cfg.num_nextn_predict_layers == 1 and cfg.mtp_layer_idx == 53 == api.MTP_LAYER_IDX
    assert cfg.prefill_cost_table == dict(pp.DEFAULT_PREFILL_COST_TABLE)
    assert "span cap=8192 A=128 kv_write=row spec=0" in cfg.describe()
    cfg.set_kv_geometry(8225, 32)  # block 32: A stays 128 (q/k 128 at C = 128 and C >= 2048), W' doubles
    assert cfg.prefill_resume_alignment == 128 and cfg.sp1_page_table_width == 1280
    small = _cfg(max_model_len=4096)  # max_model_len below the cap: the cap clamps to the last bucket
    assert small.max_prefill_span == 4096 and small.prefill_span_buckets[-1] == 4096
    assert small.sp1_page_table_width == 128


def test_features_config_env_and_validation(monkeypatch):
    monkeypatch.setenv("MOTIF3_PREFILL_MAX_BUCKET", "32768")  # restores the draft-1 bucket set
    cfg = _cfg()
    assert cfg.max_prefill_span == 32768 and cfg.prefill_span_buckets == cfg.prefill_buckets
    assert cfg.sp1_page_table_width == 1024
    monkeypatch.setenv("MOTIF3_PREFILL_MAX_BUCKET", "5000")
    with pytest.raises(ValueError, match="power of two"):
        _cfg()
    monkeypatch.delenv("MOTIF3_PREFILL_MAX_BUCKET")
    assert _cfg(prefill_span_cap=4096).prefill_span_buckets[-1] == 4096
    for bad in (
        dict(spec_tokens=2),
        dict(prefill_span_cap=3000),
        dict(prefill_span_cap=64),
        dict(prefill_cost_table={}),
        dict(prefill_cost_table={128: 0.0}),
        dict(prefill_sp1_s_per_row_key=-1.0),
        dict(kv_replicated_decode=None),
        dict(spec_tokens=1, num_nextn_predict_layers=0),
    ):
        with pytest.raises((ValueError, TypeError)):
            _cfg(**bad)


@pytest.mark.parametrize(
    "kvr, spec, mode", [(False, 0, "row"), (False, 1, "row_split"), (True, 0, "all"), (True, 1, "all_split")]
)
def test_kv_write_modes(kvr, spec, mode):
    """KV-R x speculation -> the decode KV-write mode (§3.5); one mode for all 53 layers + the MTP layer."""
    assert api.kv_write_mode(kvr, bool(spec)) == mode and mode in api.KV_WRITE_MODES
    cfg = _cfg(kv_replicated_decode=kvr, spec_tokens=spec)
    assert cfg.kv_write_mode == mode and cfg.mtp_kv_layers == spec and cfg.kv_pool_layers == 53 + spec
    s = api.GeneratorSettings(prefix_caching=kvr, spec_tokens=spec)  # auto: KV-R iff prefix caching
    assert s.kv_write_mode == mode and s.kv_replicated == kvr and s.mtp_kv_layers == spec


def test_kv_pool_bytes_with_mtp():
    """The MTP cache is one more [N, 1, 64, 576] layer: +612 B per token per chip in bfp8 (§1.2: the spec plan's
    extra_bytes_per_token)."""
    cfg = _cfg(spec_tokens=1)
    assert (
        cfg.kv_pool_bytes_per_chip() == 54 * 4129 * 2 * 18 * 1088 == api.kv_cache_bytes_per_chip(4129, 64, 54, "bfp8")
    )
    assert cfg.kv_pool_bytes_per_chip() - cfg.kv_cache_bytes_per_chip() == 4129 * 64 * 612
    assert _cfg(spec_tokens=1, kv_cache_dtype="bf16").kv_pool_bytes_per_chip() == api.kv_cache_bytes_per_chip(
        4129, 64, 54, "bf16"
    )


def test_mtp_layer_spec_matches_reference():
    """``model.mtp_layers.0``: SWA "all" mode, window 129, plain RoPE, scale 192^-0.5, dense MLP (reference
    ``MotifMTP`` = ``GDLAttention(args, 53, swa=True)``); cfg.layer(53) does not exist, hence the explicit spec."""
    cfg = _cfg()
    spec = cfg.mtp_layer_spec()
    assert spec == LayerSpec(
        idx=53, is_global=False, is_moe=False, window=129, softmax_scale=192**-0.5, rope_kind="plain"
    )
    assert spec.is_swa and spec.is_dense and spec.attn_kind == "swa" and spec.sliding_window_size == 129
    assert abs(spec.softmax_scale - 0.07216878) < 1e-8
    assert cfg.polynorm_output_scale_for_layer(cfg.mtp_layer_idx) == 0.5
    assert _cfg(num_layers=4).mtp_layer_spec().idx == 53  # the reference index, whatever a truncated run uses
    with pytest.raises(IndexError):
        cfg.layer(53)
    with pytest.raises(ValueError):
        _cfg(num_nextn_predict_layers=0).mtp_layer_spec()
    ref_cfg = pytest.importorskip("models.demos.motif3.reference.config")
    args = ref_cfg.MotifArgs.from_hf_config(HF_META)
    assert spec.window == args.effective_sliding_window
    assert spec.softmax_scale == pytest.approx(args.softmax_scale(53, swa=True), rel=1e-15)
    assert (spec.rope_kind == "yarn") == args.uses_yarn(53, swa=True)
    assert cfg.polynorm_output_scale_for_layer(53) == args.polynorm_output_scale_for_layer(53)
    assert cfg.num_nextn_predict_layers == int(args.num_nextn_predict_layers) == 1


def test_resumed_prefill_program_configs():
    """sp1 global: flexible chunked SDPA with gate G9's per-bucket q/k (128/128 at C = 128 and C >= 2048, 64/64 at
    256-1024; the table is prefill_plan's); sp1 SWA: the G2 square config over 128 + C."""
    mesh = _FakeMesh()
    cfg = _cfg()
    assert SP1_GLOBAL_CHUNKS == pp.DEFAULT_SP1_GLOBAL_CHUNKS
    for C in cfg.prefill_span_buckets:
        qk = (64, 64) if 256 <= C <= 1024 else (128, 128)
        assert cfg.sp1_global_chunks(C) == qk == pp.sp1_global_qk(C)
        want = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(12, 10), q_chunk_size=qk[0], k_chunk_size=qk[1],
            exp_approx_mode=False,
        )  # fmt: skip
        assert repr(cfg.resumed_prefill_pc("global", C)) == repr(want) == repr(resumed_prefill_pc("global", C, mesh))
        assert repr(cfg.resumed_prefill_pc(0, C)) == repr(want)  # layer 0 is global
        swa = repr(sdpa_prefill_pc("swa", mesh, seq_len=128 + C))
        assert repr(cfg.resumed_prefill_pc("swa", C)) == swa == repr(cfg.resumed_prefill_pc(cfg.layer(1), C))
    for upto, (q, k) in SP1_GLOBAL_CHUNKS:  # tile multiples, <= 128 (256/128 overflows L1), divide A
        assert q % 32 == 0 and k % 32 == 0 and q <= 128 and k <= 128
        assert cfg.prefill_resume_alignment % q == 0 and cfg.prefill_resume_alignment % k == 0
    with pytest.raises(ValueError):
        sp1_global_chunks(100)
    with pytest.raises(ValueError):
        sp1_global_chunks(65536)
    r10 = ((512, (64, 64)), (32768, (128, 128)))  # review R10's per-bucket candidate -> A = 128, budget 8064
    assert sp1_global_chunks(512, r10) == (64, 64) and sp1_global_chunks(1024, r10) == (128, 128)
    # a bf16 latent cache: 64/64 everywhere (A = 64); the table follows the cache dtype (name, ttnn dtype or config)
    b16 = _cfg(kv_cache_dtype="bf16")
    assert b16.sp1_global_chunk_table() == pp.SP1_GLOBAL_CHUNKS_BF16_KV and b16.prefill_resume_alignment == 64
    pc64 = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(12, 10), q_chunk_size=64, k_chunk_size=64, exp_approx_mode=False
    )
    for C in cfg.prefill_span_buckets:
        assert b16.sp1_global_chunks(C) == (64, 64) == cfg.sp1_global_chunks(C, kv_dtype=ttnn.bfloat16)
        assert repr(cfg.resumed_prefill_pc("global", C, kv_dtype="bf16")) == repr(pc64)
        assert cfg.sp1_global_chunks(C, kv_dtype=ttnn.bfloat8_b) == cfg.sp1_global_chunks(C)
    with pytest.raises(ValueError, match="KV cache dtype"):
        cfg.sp1_global_chunks(128, kv_dtype=ttnn.float32)


def test_cfg_plan_prefill_row_matches_free_function():
    cfg = _cfg()
    for s, e in ((0, 1000), (1348, 3000), (6976, 9000), (0, 16736), (32704, 32768), (64, 900)):
        p = cfg.plan_prefill_row(s, e)
        q = pp.plan_prefill_row(s, e, block_size=64, align=128, buckets=api.prefill_buckets(32768), span_cap=8192)
        assert p == q
        assert pp.plan_cost(p, cfg.prefill_cost) == pytest.approx(pp.plan_cost(q))
    with pytest.raises(ValueError):
        cfg.plan_prefill_row(0, 32769)
    assert cfg.prefill_cost(2048) == pytest.approx(1.55) and cfg.prefill_cost(128, 1000) > cfg.prefill_cost(128, 0)
    flat = _cfg(prefill_cost_table={b: 1.0 for b in cfg.prefill_buckets}, prefill_sp1_s_per_row_key=0.0)
    assert [(c.start, c.bucket) for c in flat.plan_prefill_row(0, 2200).chunks] == [(0, 4096)]  # re-measured table
    cfg.set_kv_geometry(8225, 32)  # the geometry allocate_kv_cache recorded wins
    p = cfg.plan_prefill_row(100, 1000)
    assert (p.block_size, p.align, p.w0, p.c0) == (32, 128, 96, 0)


def test_from_settings_features(monkeypatch):
    raw = json.load(open(f"{HF_META}/config.json"))
    s = api.GeneratorSettings(prefix_caching=True, chunked_prefill=True, spec_tokens=1, prefill_span_cap=4096)
    cfg = MotifTTConfig.from_settings(s, mesh_shape=(4, 8), hf_config=raw)
    assert cfg.kv_replicated_decode and cfg.spec_tokens == 1 and cfg.kv_write_mode == "all_split"
    assert cfg.max_prefill_span == 4096 and cfg.kv_pool_layers == 54
    d1 = MotifTTConfig.from_settings(api.GeneratorSettings(), mesh_shape=(4, 8), hf_config=raw)
    assert (d1.kv_write_mode, d1.max_prefill_span, d1.spec_tokens) == ("row", 8192, 0)
    monkeypatch.setenv("MOTIF3_PREFILL_MAX_BUCKET", "16384")  # env when the settings carry no cap; settings win
    assert (
        MotifTTConfig.from_settings(api.GeneratorSettings(), mesh_shape=(4, 8), hf_config=raw).max_prefill_span == 16384
    )
    assert MotifTTConfig.from_settings(s, mesh_shape=(4, 8), hf_config=raw).max_prefill_span == 4096
    duck = SimpleNamespace(
        num_layers=2, max_seq_len=4096, max_batch_size=4, kv_cache_dtype="bfp8"
    )  # pre-feature object
    old = MotifTTConfig.from_settings(duck, mesh_shape=(4, 8), hf_config=raw)
    assert old.kv_write_mode == "row" and old.spec_tokens == 0


# ---- generator_api: settings, rows, spec types, the ABC defaults ---------------------------------------------
def _preq(*, lane, n, start=0, width=512):
    pt = torch.zeros(width, dtype=torch.int32)
    pt[: -(-n // 64)] = torch.arange(1, -(-n // 64) + 1, dtype=torch.int32)
    return api.PrefillRequest(lane=lane, tokens=torch.zeros(n, dtype=torch.int32), page_table=pt, start=start)


def test_generator_settings_features():
    s = api.GeneratorSettings()
    assert (s.chunked_prefill, s.prefix_caching, s.max_num_batched_tokens, s.long_prefill_token_threshold) == (
        False,
        False,
        None,
        0,
    )
    assert (s.spec_tokens, s.kv_replicated_decode, s.prefill_span_cap, s.packed_prefill, s.spec_verify) == (
        0,
        None,
        None,
        False,
        "packed",
    )
    assert not s.resumed_prefill and not s.spec_decode and not s.kv_replicated and s.kv_write_mode == "row"
    assert s.resolved_prefill_span_cap(True) == 8192 and s.resolved_prefill_span_cap(False) == 32768
    p = api.GeneratorSettings(
        chunked_prefill=True,
        prefix_caching=True,
        max_num_batched_tokens=8128,
        long_prefill_token_threshold=8128,
        spec_tokens=1,
    )
    assert p.resumed_prefill and p.kv_replicated and p.kv_write_mode == "all_split" and p.mtp_kv_layers == 1
    assert api.GeneratorSettings(chunked_prefill=True).kv_write_mode == "row"  # chunking alone needs no KV-R
    assert api.GeneratorSettings(kv_replicated_decode=True).kv_write_mode == "all"  # forced on
    assert api.GeneratorSettings(max_seq_len=4096).resolved_prefill_span_cap(True) == 4096
    assert api.GeneratorSettings(prefill_span_cap=32768).resolved_prefill_span_cap(True) == 32768
    assert api.GeneratorSettings(prefill_span_cap=32768).resolved_prefill_span_cap(False) == 32768
    assert api.GeneratorSettings(max_seq_len=6144, prefill_span_cap=6144).resolved_prefill_span_cap(True) == 6144
    with pytest.raises(ValueError, match="resumed prefill"):
        api.GeneratorSettings(prefill_span_cap=8192).resolved_prefill_span_cap(False)
    for bad, exc in (
        (dict(prefix_caching=True, kv_replicated_decode=False), ValueError),  # stale cross-row KV (§3.4)
        (dict(spec_tokens=2), ValueError),
        (dict(spec_tokens=-1), ValueError),
        (dict(max_num_batched_tokens=0), ValueError),
        (dict(long_prefill_token_threshold=-1), ValueError),
        (dict(prefill_span_cap=5000), ValueError),
        (dict(prefill_span_cap=64), ValueError),
        (dict(spec_verify="tall"), ValueError),
        (dict(chunked_prefill="yes"), TypeError),
        (dict(kv_replicated_decode="auto"), TypeError),
    ):
        with pytest.raises(exc):
            api.GeneratorSettings(**bad)


def test_generator_settings_from_env_serving():
    """The bridge's captured vLLM scheduler config (``serving``) + the Motif feature environment."""
    hf = SimpleNamespace(num_hidden_layers=53)
    serving = dict(
        block_size=64,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8128,
        long_prefill_token_threshold=8128,
        enable_prefix_caching=True,
        prefix_match_unit=None,
        spec_tokens=1,
    )
    assert set(serving) == set(api.SERVING_KEYS)
    s = api.GeneratorSettings.from_env(hf, max_batch_size=32, max_seq_len=32768, environ={}, serving=serving)
    assert (s.block_size, s.chunked_prefill, s.prefix_caching, s.max_num_batched_tokens) == (64, True, True, 8128)
    assert (s.long_prefill_token_threshold, s.spec_tokens) == (8128, 1)
    assert s.kv_replicated_decode is None and s.kv_replicated and s.kv_write_mode == "all_split"
    assert s.prefill_span_cap is None and not s.packed_prefill and s.spec_verify == "packed"
    kw = dict(max_batch_size=32, max_seq_len=32768)
    assert api.GeneratorSettings.from_env(hf, block_size=32, environ={}, serving=serving, **kw).block_size == 32
    d1 = api.GeneratorSettings.from_env(hf, environ={}, **kw)
    assert not d1.resumed_prefill and not d1.spec_decode and d1.kv_write_mode == "row" and d1.block_size is None
    env = {
        "MOTIF3_KV_REPLICATED_DECODE": "1",
        "MOTIF3_PREFILL_MAX_BUCKET": "16384",
        "MOTIF3_PACKED_PREFILL": "on",
        "MOTIF3_SPEC_VERIFY": "wide",
    }
    s = api.GeneratorSettings.from_env(hf, environ=env, **kw)
    assert (s.kv_replicated_decode, s.prefill_span_cap, s.packed_prefill, s.spec_verify) == (True, 16384, True, "wide")
    assert s.kv_write_mode == "all"
    with pytest.raises(ValueError, match="KV-R"):
        api.GeneratorSettings.from_env(hf, environ={"MOTIF3_KV_REPLICATED_DECODE": "0"}, serving=serving, **kw)
    with pytest.raises(ValueError, match="unknown serving"):
        api.GeneratorSettings.from_env(hf, environ={}, serving={"max_num_batched_token": 8128}, **kw)


def test_feature_env_parsers():
    for name in api.FEATURE_SWITCHES:
        assert api.feature_switch_from_env(name, {}) is True
        assert api.feature_switch_from_env(name, {}, default=False) is False
        assert api.feature_switch_from_env(name, {name: " 0 "}) is False
        assert api.feature_switch_from_env(name, {name: "Yes"}) is True
        with pytest.raises(ValueError):
            api.feature_switch_from_env(name, {name: "ture"})  # a typo raises instead of turning a feature off
    with pytest.raises(ValueError):
        api.feature_switch_from_env("MOTIF3_NOPE", {})
    vals = [api.kv_replicated_decode_from_env({"MOTIF3_KV_REPLICATED_DECODE": v}) for v in ("auto", "", "1", "off")]
    assert vals == [None, None, True, False] and api.kv_replicated_decode_from_env({}) is None
    with pytest.raises(ValueError):
        api.kv_replicated_decode_from_env({"MOTIF3_KV_REPLICATED_DECODE": "maybe"})
    assert api.prefill_span_cap_from_env({}) is None
    assert api.prefill_span_cap_from_env({"MOTIF3_PREFILL_MAX_BUCKET": "32768"}) == 32768
    for bad in ("8000", "65536", "64", "-1", "x"):
        with pytest.raises(ValueError):
            api.prefill_span_cap_from_env({"MOTIF3_PREFILL_MAX_BUCKET": bad})
    assert api.packed_prefill_from_env({}) is False and api.spec_verify_from_env({}) == "packed"
    with pytest.raises(ValueError):
        api.spec_verify_from_env({"MOTIF3_SPEC_VERIFY": "both"})
    assert api.check_prefill_span_cap(6144, 6144) == 6144
    with pytest.raises(ValueError):
        api.check_prefill_span_cap(6144, 32768)


def test_prefill_request_start():
    r = _preq(lane=3, n=1000, start=640)
    assert (r.start, r.end, r.seq_len, r.num_new_tokens, r.resumed) == (640, 1000, 1000, 360, True)
    r0 = _preq(lane=0, n=5)
    assert r0.start == 0 and not r0.resumed and r0.num_new_tokens == 5
    for bad in (-1, 1000, 1001):
        with pytest.raises(ValueError, match="start"):
            _preq(lane=0, n=1000, start=bad)
    legacy = api.PrefillRequest(4, torch.zeros(9, dtype=torch.int32), torch.zeros(512, dtype=torch.int32))
    assert legacy.start == 0  # draft-1 positional construction keeps working


def test_spec_decode_types():
    pos = torch.full((32,), -1, dtype=torch.int32)
    pos[[0, 9, 17]] = torch.tensor([100, 5000, 63], dtype=torch.int32)
    tok = torch.zeros(32, dtype=torch.int32)
    tok[[0, 9, 17]] = torch.tensor([11, 12, 13], dtype=torch.int32)
    d = torch.full((32,), -1, dtype=torch.int32)
    d[[0, 17]] = torch.tensor([7, 0], dtype=torch.int32)  # token id 0 is a valid draft
    pt = torch.zeros(32, 512, dtype=torch.int32)
    b = api.SpecDecodeBatch(tokens=tok, positions=pos, draft_tokens=d, page_table=pt)
    assert b.num_drafts == 2 and b.is_verify and b.page_table_width == 512
    assert b.idle_lanes == tuple(lane for lane in range(32) if lane not in (0, 9, 17))
    assert torch.equal(b.has_draft, d >= 0) and torch.equal(b.active, pos >= 0)
    plain = api.SpecDecodeBatch.from_decode_batch(api.DecodeBatch(tokens=tok, positions=pos, page_table=pt))
    assert not plain.is_verify and plain.num_drafts == 0 and bool((plain.draft_tokens == -1).all())
    assert torch.equal(b.anchors().positions, pos) and isinstance(b.anchors(), api.DecodeBatch)
    bad = d.clone()
    bad[1] = 5  # lane 1 is inactive
    with pytest.raises(ValueError, match="inactive lane"):
        api.SpecDecodeBatch(tokens=tok, positions=pos, draft_tokens=bad, page_table=pt)
    for kw, exc in (
        (dict(draft_tokens=torch.full((32,), -2, dtype=torch.int32)), ValueError),
        (
            dict(
                positions=torch.full((32,), -2, dtype=torch.int32),
                draft_tokens=torch.full((32,), -1, dtype=torch.int32),
            ),
            ValueError,
        ),
        (dict(draft_tokens=d.long()), TypeError),
        (dict(tokens=tok[:31]), ValueError),
        (dict(page_table=pt[:31]), ValueError),
    ):
        args = dict(tokens=tok, positions=pos, draft_tokens=d, page_table=pt)
        args.update(kw)
        with pytest.raises(exc):
            api.SpecDecodeBatch(**args)
    am = torch.zeros(32, 2, dtype=torch.int32)
    r = api.SpecDecodeResult(logits=None, argmax=am, mtp_argmax=am.clone())
    assert api.check_spec_result("g", r, want_logits=False, vocab_size=64) is r
    with pytest.raises(ValueError, match="want_logits=True"):
        api.check_spec_result("g", r, want_logits=True, vocab_size=64)
    rl = api.SpecDecodeResult(logits=torch.zeros(32, 64, dtype=torch.bfloat16), argmax=am, mtp_argmax=am)
    assert api.check_spec_result("g", rl, want_logits=True, vocab_size=64) is rl
    with pytest.raises(ValueError, match="want_logits=False"):
        api.check_spec_result("g", rl, want_logits=False, vocab_size=64)
    with pytest.raises(ValueError):
        api.check_spec_result("g", rl, want_logits=True, vocab_size=65)
    with pytest.raises(TypeError):
        api.check_spec_result("g", (am, am), want_logits=False, vocab_size=64)
    with pytest.raises(ValueError):
        api.SpecDecodeResult(logits=None, argmax=torch.zeros(32, 3, dtype=torch.int32), mtp_argmax=am)
    with pytest.raises(TypeError):
        api.SpecDecodeResult(logits=None, argmax=am.long(), mtp_argmax=am)
    with pytest.raises(TypeError):
        api.SpecDecodeResult(logits=torch.zeros(32, 64, dtype=torch.int32), argmax=am, mtp_argmax=am)


class _TinyGenerator(api.MotifGenerator):
    """A draft-1-style generator: only the abstract members (every feature member keeps its default)."""

    def __init__(self, vocab=64):
        self._vocab = vocab
        self.calls = []

    @classmethod
    def create(cls, *, hf_config, mesh_device, settings):
        return cls()

    @property
    def num_layers(self):
        return 2

    @property
    def vocab_size(self):
        return self._vocab

    def allocate_kv_cache(self, *, num_blocks, block_size, num_layers):
        return "pool"

    def prefill_forward(self, request, *, kv_cache, enable_trace=False):
        self.calls.append((request.lane, request.seq_len, request.start, enable_trace))
        out = torch.zeros(self._vocab)
        out[request.seq_len % self._vocab] = 1.0
        return out

    def decode_forward(self, batch, *, kv_cache, enable_trace):
        return torch.zeros(api.NUM_LANES, self._vocab)

    def warmup_prefill(self, *, kv_cache, enable_trace):
        return None

    def warmup_decode(self, *, kv_cache, enable_trace, page_table_width):
        return None


class _ResumedGenerator(_TinyGenerator):
    supports_resumed_prefill = True
    prefill_alignment = 64
    max_prefill_span = 8192
    supports_spec_decode = True


def test_generator_feature_defaults_and_batch_prefill():
    g = _TinyGenerator()
    assert not g.supports_resumed_prefill and g.prefill_alignment == 0 and not g.supports_spec_decode
    assert g.max_prefill_span == g.max_prefill_len == api.MAX_CONTEXT
    out = g.prefill_forward_batch([_preq(lane=5, n=10), _preq(lane=1, n=3), _preq(lane=30, n=7)], kv_cache="pool")
    assert out.shape == (3, 64) and [int(r.argmax()) for r in out] == [10, 3, 7]
    assert g.calls == [(5, 10, 0, False), (1, 3, 0, False), (30, 7, 0, False)]  # input order: no row reads the cache
    g.prefill_forward_batch([_preq(lane=0, n=4)], kv_cache="pool", enable_trace=True)
    assert g.calls[-1] == (0, 4, 0, True)
    with pytest.raises(NotImplementedError, match="resume"):
        g.prefill_forward_batch([_preq(lane=0, n=4), _preq(lane=1, n=200, start=64)], kv_cache="pool")
    assert len(g.calls) == 4  # refused before any row ran
    with pytest.raises(ValueError, match="distinct lanes"):
        g.prefill_forward_batch([_preq(lane=2, n=5), _preq(lane=2, n=6)], kv_cache="pool")
    with pytest.raises(ValueError, match="no rows"):
        g.prefill_forward_batch([], kv_cache="pool")
    with pytest.raises(TypeError):
        g.prefill_forward_batch([object()], kv_cache="pool")
    with pytest.raises(ValueError, match="at most"):
        api.check_prefill_batch([_preq(lane=i % 32, n=4) for i in range(33)])
    batch = api.SpecDecodeBatch.from_decode_batch(
        api.DecodeBatch(
            tokens=torch.zeros(32, dtype=torch.int32),
            positions=torch.full((32,), -1, dtype=torch.int32),
            page_table=torch.zeros(32, 8, dtype=torch.int32),
        )
    )
    with pytest.raises(NotImplementedError):
        g.decode_forward_spec(batch, kv_cache="pool", enable_trace=False, want_logits=True)


def test_check_generator_features():
    """A feature vLLM enabled must be supported (§1.5); the resumed geometry must be consistent."""
    d1, res = _TinyGenerator(), _ResumedGenerator()
    plain = api.GeneratorSettings(block_size=64)
    api.check_generator_features(d1, plain)
    api.check_generator_features(res, plain)
    for s, what in (
        (api.GeneratorSettings(chunked_prefill=True), "chunked prefill"),
        (api.GeneratorSettings(prefix_caching=True), "prefix caching"),
    ):
        with pytest.raises(ValueError, match=what):
            api.check_generator_features(d1, s)
        api.check_generator_features(res, s)
    with pytest.raises(ValueError, match="speculative"):
        api.check_generator_features(d1, api.GeneratorSettings(spec_tokens=1))
    api.check_generator_features(res, api.GeneratorSettings(spec_tokens=1))

    class BadAlign(_ResumedGenerator):
        prefill_alignment = 96

    with pytest.raises(ValueError, match="prefill_alignment"):
        api.check_generator_features(BadAlign(), plain)

    class SplitsWithoutResume(_TinyGenerator):
        max_prefill_span = 8192

    with pytest.raises(ValueError, match="no resumed prefill"):
        api.check_generator_features(SplitsWithoutResume(), plain)


# ======================================================================================================
# P5 (packed prefill) and T64 (full-batch speculative verify): the settings / config fields of step C1a
# (docs/p5_t64/P5_T64_DESIGN.md §8.3, §8.5; host tests of §7.1)
# ======================================================================================================
def test_p5_t64_contract_constants():
    """The frozen vocabulary of both features: verify modes, the T64 row counts, the packed segment sizes / batches /
    pass kinds / pk1 tail variants, the drafting-policy constants."""
    assert api.SPEC_VERIFY_MODES == ("packed", "wide", "auto") and api.WIDE_SPEC_VERIFY_MODES == ("wide", "auto")
    assert api.WIDE_ROWS == 2 * api.NUM_LANES == 64 == api.NUM_DP_GROUPS * api.WIDE_ROWS_PER_GROUP
    assert api.WIDE_ROWS_PER_GROUP == 2 * api.LANES_PER_GROUP == 16 <= 32  # [8 anchors | 8 drafts]: one tile row
    assert api.WIDE_MIN_LANES_NEVER == api.NUM_LANES + 1 == 33
    assert (api.DEFAULT_SPEC_ALPHA_PRIOR, api.SPEC_ALPHA_PRIOR_WEIGHT) == (0.85, 64)
    assert api.PACKED_PASS_KINDS == ("pk0", "pk1") and api.PK1_TAIL_VARIANTS == ("shared", "distinct")
    assert api.PACK_SEG_BUCKETS == (64, 128, 256, 512, 1024) and api.PACK_SP1_SEG_BUCKETS == (128, 256, 512, 1024)
    assert set(api.PACK_SP1_SEG_BUCKETS) <= set(api.PACK_SEG_BUCKETS)
    assert api.PACK_BATCHES == (2, 4, 8, 16, 32) and max(api.PACK_BATCHES) == api.NUM_LANES  # rows of one call
    for s in api.PACK_SEG_BUCKETS + api.PACK_BATCHES:
        assert s & (s - 1) == 0, s  # powers of two: T = B * S is a prefill bucket
    assert api.DEFAULT_PACKED_PREFILL_MAX_SEG == max(api.PACK_SEG_BUCKETS) == 1024
    assert api.DEFAULT_PACKED_PREFILL_MAX_TOKENS == api.DEFAULT_PREFILL_SPAN_CAP == 8192
    assert api.PACKED_WARMUP_MODES == ("attention", "full") and api.DEFAULT_PACKED_WARMUP == "attention"


def test_p5_t64_settings_defaults_and_validation():
    s = api.GeneratorSettings()
    assert (s.packed_prefill, s.packed_prefill_max_seg, s.packed_prefill_max_tokens, s.packed_prefill_pk1) == (
        False,
        1024,
        8192,
        True,
    )
    assert (s.packed_warmup, s.spec_verify, s.wide_min_lanes, s.spec_alpha_prior) == ("attention", "packed", None, 0.85)
    for ok in (
        dict(spec_verify="auto", spec_tokens=1),
        dict(spec_verify="wide"),  # no effect without speculation, but not an error
        dict(packed_prefill=True, packed_prefill_max_seg=64, packed_prefill_max_tokens=128, packed_prefill_pk1=False),
        dict(packed_prefill_max_tokens=32768, packed_warmup="full"),
        dict(wide_min_lanes=1),
        dict(wide_min_lanes=33),  # never
        dict(spec_alpha_prior=0.0),
        dict(spec_alpha_prior=1),
    ):
        api.GeneratorSettings(**ok)
    for bad, exc in (
        (dict(spec_verify="Auto"), ValueError),  # the settings take the parsed (lower-case) mode
        (dict(spec_verify="both"), ValueError),
        (dict(packed_prefill_max_seg=2048), ValueError),  # beyond the segment sizes gate G15 validates
        (dict(packed_prefill_max_seg=100), ValueError),
        (dict(packed_prefill_max_tokens=3000), ValueError),
        (dict(packed_prefill_max_tokens=64), ValueError),
        (dict(packed_prefill_max_tokens=65536), ValueError),
        (dict(packed_prefill_pk1="yes"), TypeError),
        (dict(packed_warmup="none"), ValueError),
        (dict(wide_min_lanes=0), ValueError),
        (dict(wide_min_lanes=34), ValueError),
        (dict(spec_alpha_prior=1.5), ValueError),
        (dict(spec_alpha_prior=-0.1), ValueError),
        (dict(spec_alpha_prior=float("nan")), ValueError),
        (dict(spec_alpha_prior=True), TypeError),
    ):
        with pytest.raises(exc):
            api.GeneratorSettings(**bad)


def test_p5_t64_env_parsing():
    """MOTIF3_PACKED_PREFILL_* / MOTIF3_PACKED_WARMUP / MOTIF3_SPEC_VERIFY=auto / MOTIF3_WIDE_MIN_LANES: parsed by
    GeneratorSettings.from_env (API server and EngineCore alike); a typo raises instead of silently changing a knob."""
    hf = SimpleNamespace(num_hidden_layers=53)
    kw = dict(max_batch_size=32, max_seq_len=32768)
    d = api.GeneratorSettings.from_env(hf, environ={}, **kw)
    assert (d.packed_prefill, d.packed_prefill_max_seg, d.packed_prefill_max_tokens, d.packed_prefill_pk1) == (
        False,
        1024,
        8192,
        True,
    )
    assert (d.packed_warmup, d.spec_verify, d.wide_min_lanes, d.spec_alpha_prior) == ("attention", "packed", None, 0.85)
    env = {
        "MOTIF3_PACKED_PREFILL": "1",
        "MOTIF3_PACKED_PREFILL_MAX_SEG": "512",
        "MOTIF3_PACKED_PREFILL_MAX_TOKENS": "4096",
        "MOTIF3_PACKED_PREFILL_PK1": "off",
        "MOTIF3_PACKED_WARMUP": " FULL ",
        "MOTIF3_SPEC_VERIFY": " Auto ",
        "MOTIF3_WIDE_MIN_LANES": "20",
    }
    s = api.GeneratorSettings.from_env(hf, environ=env, serving=dict(spec_tokens=1), **kw)
    assert (s.packed_prefill, s.packed_prefill_max_seg, s.packed_prefill_max_tokens, s.packed_prefill_pk1) == (
        True,
        512,
        4096,
        False,
    )
    assert (s.packed_warmup, s.spec_verify, s.wide_min_lanes, s.spec_alpha_prior) == ("full", "auto", 20, 0.85)
    # the parsers one by one: unset / blank = the default
    assert api.packed_prefill_max_seg_from_env({}) == 1024 and api.packed_prefill_max_tokens_from_env({}) == 8192
    assert api.packed_prefill_pk1_from_env({}) is True and api.packed_warmup_from_env({}) == "attention"
    assert api.wide_min_lanes_from_env({}) is None and api.spec_verify_from_env({"MOTIF3_SPEC_VERIFY": " "}) == "packed"
    assert api.packed_warmup_from_env({"MOTIF3_PACKED_WARMUP": " "}) == "attention"
    assert api.spec_verify_from_env({"MOTIF3_SPEC_VERIFY": "WIDE"}) == "wide"
    assert [api.packed_prefill_max_seg_from_env({"MOTIF3_PACKED_PREFILL_MAX_SEG": str(v)}) for v in (64, 1024)] == [
        64,
        1024,
    ]
    assert api.packed_prefill_pk1_from_env({"MOTIF3_PACKED_PREFILL_PK1": "0"}) is False
    assert api.wide_min_lanes_from_env({"MOTIF3_WIDE_MIN_LANES": "33"}) == 33
    for name, parser, bads in (
        ("MOTIF3_PACKED_PREFILL_MAX_SEG", api.packed_prefill_max_seg_from_env, ("2048", "96", "-1", "x", "1e3")),
        ("MOTIF3_PACKED_PREFILL_MAX_TOKENS", api.packed_prefill_max_tokens_from_env, ("5000", "64", "65536", "x")),
        ("MOTIF3_PACKED_PREFILL_PK1", api.packed_prefill_pk1_from_env, ("ture", "2")),
        ("MOTIF3_PACKED_WARMUP", api.packed_warmup_from_env, ("none", "attn")),
        ("MOTIF3_SPEC_VERIFY", api.spec_verify_from_env, ("both", "t64")),
        ("MOTIF3_WIDE_MIN_LANES", api.wide_min_lanes_from_env, ("0", "34", "x", "-5")),
    ):
        for bad in bads:
            with pytest.raises(ValueError, match=name):
                parser({name: bad})
        with pytest.raises(ValueError, match=name):
            api.GeneratorSettings.from_env(hf, environ={name: bads[0]}, **kw)


def test_tt_cache_policy_settings_and_env():
    """MOTIF3_TT_CACHE_POLICY -> GeneratorSettings.tt_cache_policy (MotifGenerator.create passes it to
    MotifModel(cache=...)): default "auto" (every existing launch unchanged), the three names of tt/model.py's
    CACHE_POLICIES, case and blanks forgiven in the environment only, anything else refused."""
    model = importlib.import_module("models.demos.motif3.tt.model")
    assert api.TT_CACHE_POLICIES == model.CACHE_POLICIES == ("auto", "write", "off")
    assert api.DEFAULT_TT_CACHE_POLICY == "auto" == inspect.signature(model.MotifModel).parameters["cache"].default
    hf = SimpleNamespace(num_hidden_layers=53)
    kw = dict(max_batch_size=32, max_seq_len=32768)
    assert api.GeneratorSettings().tt_cache_policy == "auto"
    assert api.GeneratorSettings.from_env(hf, environ={}, **kw).tt_cache_policy == "auto"
    assert api.tt_cache_policy_from_env({}) == "auto"
    assert api.tt_cache_policy_from_env({"MOTIF3_TT_CACHE_POLICY": " "}) == "auto"
    for policy in api.TT_CACHE_POLICIES:
        assert api.GeneratorSettings(tt_cache_policy=policy).tt_cache_policy == policy
        assert api.check_tt_cache_policy(policy) == policy
        assert model.normalize_cache_policy(policy) == policy
        for raw in (policy, f" {policy.upper()} ", policy.capitalize()):
            assert api.tt_cache_policy_from_env({"MOTIF3_TT_CACHE_POLICY": raw}) == policy
        s = api.GeneratorSettings.from_env(hf, environ={"MOTIF3_TT_CACHE_POLICY": policy}, **kw)
        assert s.tt_cache_policy == policy
    for bad in ("Write", "read", "on", "1", True, None, ""):  # the settings take the parsed (lower-case) name only
        with pytest.raises(ValueError, match="MOTIF3_TT_CACHE_POLICY"):
            api.GeneratorSettings(tt_cache_policy=bad)
    for bad in ("read", "readonly", "1", "true", "rw", "auto,write"):
        with pytest.raises(ValueError, match="MOTIF3_TT_CACHE_POLICY"):
            api.tt_cache_policy_from_env({"MOTIF3_TT_CACHE_POLICY": bad})
        with pytest.raises(ValueError, match="MOTIF3_TT_CACHE_POLICY"):
            api.GeneratorSettings.from_env(hf, environ={"MOTIF3_TT_CACHE_POLICY": bad}, **kw)


def test_wide_step_ratio_override():
    """MOTIF3_WIDE_STEP_RATIO -> GeneratorSettings.wide_step_ratio -> MotifTTConfig.from_settings: unset keeps exactly
    today's config (r = DEFAULT_WIDE_STEP_RATIO = 1.13), a value moves r and with it c*, and the check is
    MotifTTConfig.validate's rule (finite, >= 1)."""
    from models.demos.motif3.tt import verify_plan as vp

    raw = json.load(open(f"{HF_META}/config.json"))
    hf = SimpleNamespace(num_hidden_layers=53)
    kw = dict(max_batch_size=32, max_seq_len=32768)
    spec = dict(prefix_caching=True, chunked_prefill=True, spec_tokens=1, spec_verify="auto")
    # unset: None in the settings, the default r, every config field as with the explicit default
    assert api.GeneratorSettings().wide_step_ratio is None and api.wide_step_ratio_from_env({}) is None
    assert api.GeneratorSettings.from_env(hf, environ={}, **kw).wide_step_ratio is None
    assert api.wide_step_ratio_from_env({"MOTIF3_WIDE_STEP_RATIO": "  "}) is None
    for extra in ({}, spec):
        unset = MotifTTConfig.from_settings(api.GeneratorSettings(**extra), mesh_shape=(4, 8), hf_config=raw)
        pinned = MotifTTConfig.from_settings(
            api.GeneratorSettings(wide_step_ratio=DEFAULT_WIDE_STEP_RATIO, **extra), mesh_shape=(4, 8), hf_config=raw
        )
        assert unset.wide_step_ratio == DEFAULT_WIDE_STEP_RATIO == 1.13 and _fields(unset) == _fields(pinned)
        assert unset.describe() == pinned.describe()
    # a value: the settings, the config, the T64 line of describe() and c* follow it
    for value, c_star in ((1.0, 17), (1.13, 19), (1.2, 20), (1.3, 22), (2, 33)):
        s = api.GeneratorSettings(wide_step_ratio=value, **spec)
        c = MotifTTConfig.from_settings(s, mesh_shape=(4, 8), hf_config=raw)
        assert c.wide_step_ratio == float(value) and f"r {float(value):g})" in c.describe()
        assert vp.crossover_lanes(api.DEFAULT_SPEC_ALPHA_PRIOR, c.wide_step_ratio) == c_star, value
        env = {"MOTIF3_WIDE_STEP_RATIO": f" {value} "}
        assert api.wide_step_ratio_from_env(env) == float(value)
        assert api.GeneratorSettings.from_env(hf, environ=env, **kw).wide_step_ratio == float(value)
    assert api.wide_step_ratio_from_env({"MOTIF3_WIDE_STEP_RATIO": "1.15e0"}) == 1.15
    # the same rule as MotifTTConfig.validate
    for value in (0.0, 0.99, 1.0, 1.13, 5.0, -1.0, float("nan"), float("inf"), float("-inf")):
        try:
            _cfg(wide_step_ratio=value)
            cfg_ok = True
        except ValueError:
            cfg_ok = False
        try:
            api.check_wide_step_ratio(value)
            api_ok = True
        except ValueError:
            api_ok = False
        assert api_ok == cfg_ok, value
    for bad, exc in ((0.99, ValueError), (float("nan"), ValueError), (float("inf"), ValueError), (True, TypeError)):
        with pytest.raises(exc, match="MOTIF3_WIDE_STEP_RATIO"):
            api.GeneratorSettings(wide_step_ratio=bad)
    for bad in ("0.99", "0", "-1.2", "nan", "inf", "x", "1,2", "1.2.3"):
        with pytest.raises(ValueError, match="MOTIF3_WIDE_STEP_RATIO"):
            api.wide_step_ratio_from_env({"MOTIF3_WIDE_STEP_RATIO": bad})
        with pytest.raises(ValueError, match="MOTIF3_WIDE_STEP_RATIO"):
            api.GeneratorSettings.from_env(hf, environ={"MOTIF3_WIDE_STEP_RATIO": bad}, **kw)


def test_smoothed_acceptance_prior():
    """Review edit R-E3: alpha-hat = (accepted + n0 a0) / (verified + n0). A fresh server (nothing verified) assumes
    a0 = 0.85, far above the T64 break-even r - 1 ~ 0.13, so a c = 32 burst drafts from its first verify step."""
    sa = api.smoothed_acceptance
    assert sa(0, 0) == api.DEFAULT_SPEC_ALPHA_PRIOR == 0.85
    assert sa(0, 0) > DEFAULT_WIDE_STEP_RATIO - 1.0
    assert sa(0, 0, prior=0.5) == 0.5 and sa(0, 0, weight=0) == 0.85  # nothing to average: the prior
    assert sa(30, 32) == pytest.approx((30 + 64 * 0.85) / 96)
    assert sa(30, 32, weight=0) == pytest.approx(30 / 32)
    assert sa(88_000, 100_000) == pytest.approx(0.88, abs=1e-4)  # many verifies: the measured rate
    assert 0.0 <= sa(0, 10_000) < 0.01 and 0.99 < sa(10_000, 10_000) <= 1.0
    lo, mid, hi = sa(0, 64), sa(32, 64), sa(64, 64)
    assert lo < mid < hi and lo == pytest.approx(0.425)  # 64 rejects halve the prior
    for bad in (dict(accepted=5, verified=4), dict(accepted=-1, verified=4), dict(accepted=0, verified=0, weight=-1)):
        with pytest.raises(ValueError):
            sa(**bad)
    with pytest.raises(ValueError):
        sa(1, 2, prior=1.2)


def test_drafts_all_lanes_contract():
    """Review edit R-E9: the bridge asks a yes / no, ``drafts_all_lanes(live_lanes, acceptance=None)``; the ABC default
    (and every generator without the 64-row trace) answers False, which keeps the bridge's idle-lane budget."""
    sig = inspect.signature(api.MotifGenerator.drafts_all_lanes)
    assert list(sig.parameters) == ["self", "live_lanes", "acceptance"]
    assert sig.parameters["acceptance"].default is None
    for g in (_TinyGenerator(), _ResumedGenerator()):
        assert g.drafts_all_lanes([0, 1, 2]) is False
        assert g.drafts_all_lanes(list(range(32)), acceptance=0.99) is False
        assert g.drafts_all_lanes([], acceptance=None) is False

    class Wide(_ResumedGenerator):
        def drafts_all_lanes(self, live_lanes, acceptance=None):
            return len(set(live_lanes)) >= 19

    assert Wide().drafts_all_lanes(list(range(19))) and not Wide().drafts_all_lanes(list(range(18)))


def test_packed_prefill_config_and_shapes(monkeypatch):
    """P5 shapes (design §3.5): 22 pk0 (T, S) and 17 pk1 (T, S), each pk1 shape with both SWA tail variants (R-E2);
    every T = B * S a span bucket <= 8192; every segment size has its SDPA programs; the settings / env knobs narrow
    the list."""
    cfg = _cfg()
    assert cfg.pack_seg_buckets == api.PACK_SEG_BUCKETS and cfg.pack_sp1_seg_buckets == api.PACK_SP1_SEG_BUCKETS
    assert (cfg.pack_max_tokens, cfg.pack_tokens_cap, cfg.pack_max_seg, cfg.pack_pk1) == (8192, 8192, 1024, True)
    shapes = cfg.packed_prefill_shapes()
    assert len(shapes) == len(set(shapes)) == 22 + 2 * 17
    pk0 = [k for k in shapes if k[0] == "pk0"]
    pk1 = [k for k in shapes if k[0] == "pk1"]
    assert shapes == tuple(pk0 + pk1)  # pk0 first, then pk1: a fixed warmup order
    assert Counter(S for _, _, S in pk0) == {64: 5, 128: 5, 256: 5, 512: 4, 1024: 3}
    assert Counter(k[2] for k in pk1) == {128: 10, 256: 10, 512: 8, 1024: 6}
    for k in shapes:
        kind, T, S = k[:3]
        assert len(k) == (3 if kind == "pk0" else 4), k
        assert T // S in api.PACK_BATCHES and T == S * (T // S), k
        assert T in cfg.prefill_span_buckets and T <= cfg.pack_tokens_cap, k
    for T, S in {(k[1], k[2]) for k in pk1}:  # both tail variants of every pk1 shape (R-E2)
        assert {k[3] for k in pk1 if k[1:3] == (T, S)} == set(api.PK1_TAIL_VARIANTS)
    # the design's scenarios (§3.3): 32 short prompts; a shared 2K prefix; the mixed step; S = 1024 at the cap
    for key in (("pk0", 2048, 64), ("pk1", 4096, 128, "shared"), ("pk1", 4096, 128, "distinct"), ("pk0", 4096, 512)):
        assert key in shapes, key
    assert ("pk0", 8192, 1024) in shapes and ("pk0", 16384, 512) not in shapes  # T <= the 8192 cap
    assert ("pk1", 128, 64, "shared") not in shapes  # no pk1 segments of 64 rows
    # every packed segment has its per-segment SDPA programs: pk0 the G2 config at S, pk1 the sp1 configs at S
    for S in cfg.pack_seg_buckets:
        for kind in ("global", "swa"):
            cfg.sdpa_prefill_pc(kind, seq_len=S)
    for S in cfg.pack_sp1_seg_buckets:
        for kind in ("global", "swa"):
            cfg.resumed_prefill_pc(kind, S)
    assert "pack S=64/128/256/512/1024 pk1 S=128/256/512/1024 T<=8192" in cfg.describe()

    # settings -> config (GEN-1 mapping)
    raw = json.load(open(f"{HF_META}/config.json"))
    s = api.GeneratorSettings(packed_prefill_max_seg=256, packed_prefill_pk1=False, packed_prefill_max_tokens=4096)
    c2 = MotifTTConfig.from_settings(s, mesh_shape=(4, 8), hf_config=raw)
    assert (c2.pack_seg_buckets, c2.pack_sp1_seg_buckets, c2.pack_max_tokens) == ((64, 128, 256), (), 4096)
    assert (c2.pack_max_seg, c2.pack_pk1, c2.pack_tokens_cap) == (256, False, 4096)
    assert len(c2.packed_prefill_shapes()) == 5 + 5 + 4 and "pk1 S=off" in c2.describe()
    assert all(k[0] == "pk0" and k[1] <= 4096 for k in c2.packed_prefill_shapes())
    c3 = MotifTTConfig.from_settings(api.GeneratorSettings(packed_prefill_max_seg=64), mesh_shape=(4, 8), hf_config=raw)
    assert (c3.pack_seg_buckets, c3.pack_sp1_seg_buckets, c3.pack_pk1) == ((64,), (), False)  # no pk1 size <= 64

    # the span cap bounds T as well: max_model_len 4096, or 6144 (a non-power-of-two last bucket)
    assert max(k[1] for k in _cfg(max_model_len=4096).packed_prefill_shapes()) == 4096
    c6 = _cfg(max_model_len=6144)
    assert c6.pack_tokens_cap == 6144 and max(k[1] for k in c6.packed_prefill_shapes()) == 4096

    # environment (configs built without settings, e.g. module tests)
    monkeypatch.setenv("MOTIF3_PACKED_PREFILL_MAX_SEG", "512")
    monkeypatch.setenv("MOTIF3_PACKED_PREFILL_MAX_TOKENS", "4096")
    monkeypatch.setenv("MOTIF3_PACKED_PREFILL_PK1", "0")
    ce = _cfg()
    assert (ce.pack_seg_buckets, ce.pack_sp1_seg_buckets, ce.pack_max_tokens) == ((64, 128, 256, 512), (), 4096)
    duck = SimpleNamespace(num_layers=2, max_seq_len=4096, max_batch_size=4, kv_cache_dtype="bfp8")  # pre-P5 object
    assert MotifTTConfig.from_settings(duck, mesh_shape=(4, 8), hf_config=raw).pack_max_tokens == 4096  # env kept
    plain = MotifTTConfig.from_settings(api.GeneratorSettings(), mesh_shape=(4, 8), hf_config=raw)
    assert plain.pack_pk1 and plain.pack_max_tokens == 8192  # the settings win over the environment
    monkeypatch.setenv("MOTIF3_PACKED_PREFILL_MAX_SEG", "2048")
    with pytest.raises(ValueError, match="MOTIF3_PACKED_PREFILL_MAX_SEG"):
        _cfg()
    for v in ("MOTIF3_PACKED_PREFILL_MAX_SEG", "MOTIF3_PACKED_PREFILL_MAX_TOKENS", "MOTIF3_PACKED_PREFILL_PK1"):
        monkeypatch.delenv(v)

    # validation
    for bad in (
        dict(pack_seg_buckets=(128, 64)),
        dict(pack_seg_buckets=(64, 64)),
        dict(pack_seg_buckets=(32,)),
        dict(pack_seg_buckets=(2048,)),
        dict(pack_sp1_seg_buckets=(64, 128)),  # pk1 S = 64 needs its own SWA square config (later)
        dict(pack_max_tokens=3000),
        dict(pack_max_tokens=64),
        dict(pack_max_tokens=65536),
    ):
        with pytest.raises(ValueError):
            _cfg(**bad)
    with pytest.raises(ValueError, match="whole 128-token blocks"):  # S = 64 would split a block of the fill table
        _cfg(kv_block_size=128)
    assert _cfg(kv_block_size=128, pack_seg_buckets=[128, 256]).pack_seg_buckets == (128, 256)  # lists normalized
    assert _cfg(pack_seg_buckets=(), pack_sp1_seg_buckets=()).packed_prefill_shapes() == ()
    small = _cfg()
    small.set_kv_geometry(8225, 32)  # block 32: every segment still whole blocks
    assert small.packed_prefill_shapes() == shapes


def test_per_core_m_matmul_builders():
    """``per_core_m`` (T64's 64 gathered rows = 2 tile rows): ``per_core_M = out_block_h = per_core_m``, out subblock
    ``h = 1`` and ``h x w <= 4`` with fp32 dest acc (8 without); ``per_core_m = 1`` is the unchanged draft-1 config."""
    for pcn in range(1, 9):
        for pcm in (1, 2, 3, 4):
            for fp32 in (True, False):
                pc = mcast1d_matmul_pc((8, 1), 8 * pcn, 1, per_core_n=pcn, per_core_m=pcm, fp32_acc=fp32)
                assert (pc.per_core_M, pc.out_block_h, pc.per_core_N, pc.out_block_w) == (pcm, pcm, pcn, pcn)
                assert pc.out_subblock_h == 1 and pcn % pc.out_subblock_w == 0
                assert pc.out_subblock_h * pc.out_subblock_w <= (4 if fp32 else 8)
    default = mcast1d_matmul_pc((12, 1), 12, 32, 128)
    assert repr(default) == repr(mcast1d_matmul_pc((12, 1), 12, 32, 128, per_core_m=1))
    assert (default.per_core_M, default.out_block_h) == (1, 1)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="per_core_m"):
            mcast1d_matmul_pc((12, 1), 12, 32, 128, per_core_m=bad)


def _g16_lite_mm1d(grid, in0_block_w, per_core_m, per_core_n, sub_h, sub_w, fuse_batch=False, act=None):
    """Verbatim ``mm1d`` of docs/p5_t64/scripts/t64_bench.py: the configs the G16-lite probe measured (T64N §5.1)."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(int(grid[0]), int(grid[1])),
        in0_block_w=int(in0_block_w),
        out_subblock_h=int(sub_h),
        out_subblock_w=int(sub_w),
        out_block_h=int(per_core_m),
        out_block_w=int(per_core_n),
        per_core_M=int(per_core_m),
        per_core_N=int(per_core_n),
        fuse_batch=bool(fuse_batch),
        fused_activation=act,
        mcast_in0=True,
    )


def test_t64_m64_decode_configs_match_the_g16_lite_probe():
    """The M = 64 decode configs (design T4, §4.2) are exactly the ones G16-lite measured: router per_core_M 2 + fused
    sigmoid on 12 x 1; gate_up per_core_M 2 on 10 x 8; down per_core_M 2 on 8 x 8 (8 x 4 by grid override); LM head
    per_core_M 2 on 12 x 9. M = 32 (``m_tiles=1``) is unchanged."""
    mm, cfg = _g16_lite_mm1d, _cfg()
    sig = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
    want64 = {
        "router": (mm((12, 1), 32, 2, 1, 1, 1), cfg.router_decode_pc(m_tiles=2), router_decode_pc(m_tiles=2)),
        "router_sigmoid": (
            mm((12, 1), 32, 2, 1, 1, 1, act=sig),
            cfg.router_decode_pc(sigmoid=True, m_tiles=2),
            router_decode_pc(12, 128, sigmoid=True, m_tiles=2),
        ),
        "gate_up": (mm((10, 8), 8, 2, 1, 1, 1), cfg.experts_gate_up_pc(m_tiles=2), experts_gate_up_pc(m_tiles=2)),
        "down_8x8": (mm((8, 8), 4, 2, 2, 1, 2), cfg.experts_down_pc(m_tiles=2), experts_down_pc(m_tiles=2)),
        "down_8x4": (
            mm((8, 4), 4, 2, 4, 1, 4),
            cfg.experts_down_pc(2, (8, 4)),
            experts_down_pc(m_tiles=2, grid=(8, 4)),
        ),
        "lm_head": (
            mm((12, 9), 16, 2, 2, 1, 2, fuse_batch=True),
            cfg.lm_head_pc("mesh", m_tiles=2),
            lm_head_pc(215, ((12, 9), 2, 16), m_tiles=2),
        ),
    }
    for name, (want, *got) in want64.items():
        for g in got:
            assert repr(g) == repr(want), name
    want32 = {
        "router_sigmoid": (mm((12, 1), 32, 1, 1, 1, 1, act=sig), cfg.router_decode_pc(sigmoid=True)),
        "gate_up": (mm((10, 8), 8, 1, 1, 1, 1), cfg.experts_gate_up_pc()),
        "down": (mm((8, 4), 4, 1, 4, 1, 4), cfg.experts_down_pc()),
        "lm_head": (mm((12, 9), 16, 1, 2, 1, 2, fuse_batch=True), cfg.lm_head_pc("mesh")),
    }
    for name, (want, got) in want32.items():
        assert repr(got) == repr(want), name
    assert repr(cfg.experts_down_pc(m_tiles=1)) == repr(cfg.experts_down_pc())  # m_tiles=1 keeps the G6 8 x 4 grid
    assert repr(cfg.lm_head_pc("mesh", m_tiles=1)) == repr(cfg.lm_head_pc("mesh"))
    tp64 = cfg.lm_head_pc("tp", m_tiles=2)  # the 'tp' vocab split at M = 64: per_core_N 8, subblock 1 x 4
    assert (tp64.per_core_M, tp64.per_core_N, tp64.out_subblock_h, tp64.out_subblock_w) == (2, 8, 1, 4)
    assert EXPERTS_DOWN_GRID_WIDE == (8, 8) and all(a <= b for a, b in zip(EXPERTS_DOWN_GRID_WIDE, cfg.compute_grid))
    assert cfg.lm_head_pc("mesh", "auto", m_tiles=2) is None


def test_spec_verify_config_and_f3_refusals(monkeypatch):
    """``spec_verify`` / ``wide_rows_per_dp`` / ``wide_step_ratio`` and the refusals the generator's ``create`` gets
    through ``MotifTTConfig.from_settings``: "wide" / "auto" need ring_gather "safe" (F3N R1; R-E5: "native" and
    "lean" refused), and "auto" with the exact-fp32 router only at a T64 row count the MoE runs it at (R-E7)."""
    cfg = _cfg()
    assert (cfg.spec_verify, cfg.wide_rows_per_dp, cfg.wide_step_ratio) == ("packed", 0, DEFAULT_WIDE_STEP_RATIO)
    assert DEFAULT_WIDE_STEP_RATIO == 1.13 and cfg.ring_gather == "safe"
    for shape in ((4, 8), (8, 4)):
        for mode in api.WIDE_SPEC_VERIFY_MODES:
            c = _cfg(mesh_shape=shape, spec_tokens=1, spec_verify=mode)
            assert c.wide_rows_per_dp == api.WIDE_ROWS_PER_GROUP == 16, (shape, mode)
            assert c.dp * c.wide_rows_per_dp == api.WIDE_ROWS
    assert _cfg(spec_tokens=1, spec_verify="packed").wide_rows_per_dp == 0
    assert _cfg(spec_verify="auto").wide_rows_per_dp == 0  # no speculation: no 64-row trace
    raw = json.load(open(f"{HF_META}/config.json"))
    s = api.GeneratorSettings(prefix_caching=True, chunked_prefill=True, spec_tokens=1, spec_verify="auto")
    c = MotifTTConfig.from_settings(s, mesh_shape=(4, 8), hf_config=raw)
    assert (c.spec_verify, c.wide_rows_per_dp, c.kv_write_mode) == ("auto", 16, "all_split")
    assert "spec=1 spec_verify=auto (T64 16 rows/DP row, r 1.13)" in c.describe() and "ring_gather=safe" in c.describe()
    plain = MotifTTConfig.from_settings(api.GeneratorSettings(), mesh_shape=(4, 8), hf_config=raw)
    assert (plain.spec_verify, plain.wide_rows_per_dp) == ("packed", 0)

    # R1 / R-E5: only "safe" with a 64-row trace
    for mode in api.WIDE_SPEC_VERIFY_MODES:
        for ring in ("native", "lean"):
            with pytest.raises(ValueError, match="ring_gather='safe'"):
                _cfg(spec_tokens=1, spec_verify=mode, ring_gather=ring)
    for ring in ("native", "lean"):  # unaffected: today's packed verify, and "auto" without speculation
        _cfg(spec_tokens=1, spec_verify="packed", ring_gather=ring)
        _cfg(spec_verify="auto", ring_gather=ring)
    monkeypatch.setenv("MOTIF3_RING_GATHER", "native")  # the path create() takes
    with pytest.raises(ValueError, match="ring_gather='safe'"):
        MotifTTConfig.from_settings(s, mesh_shape=(4, 8), hf_config=raw)
    monkeypatch.delenv("MOTIF3_RING_GATHER")

    # R-E7: "auto" + the exact-fp32 router is accepted since tt/moe.py runs it at the T64 step's M = 64 (D1); a T64 row
    # count missing from ROUTER_EXACT_FP32_DECODE_ROWS is still refused
    assert ROUTER_EXACT_FP32_DECODE_ROWS == (32, 64)
    c = _cfg(spec_tokens=1, spec_verify="auto", router_logits="exact_fp32")
    assert (c.spec_verify, c.router_logits, c.wide_rows_per_dp) == ("auto", "exact_fp32", 16)
    monkeypatch.setenv("MOTIF3_ROUTER_LOGITS", "exact_fp32")
    c = MotifTTConfig.from_settings(s, mesh_shape=(4, 8), hf_config=raw)
    assert (c.spec_verify, c.router_logits, c.wide_rows_per_dp) == ("auto", "exact_fp32", 16)
    monkeypatch.delenv("MOTIF3_ROUTER_LOGITS")
    mc = importlib.import_module("models.demos.motif3.tt.model_config")
    monkeypatch.setattr(mc, "ROUTER_EXACT_FP32_DECODE_ROWS", (32,))
    with pytest.raises(ValueError, match="lacks the 64-row T64 step"):
        _cfg(spec_tokens=1, spec_verify="auto", router_logits="exact_fp32")
    monkeypatch.setattr(mc, "ROUTER_EXACT_FP32_DECODE_ROWS", ROUTER_EXACT_FP32_DECODE_ROWS)
    _cfg(spec_tokens=1, spec_verify="wide", router_logits="exact_fp32")  # one trace: T64 rows only meet T64 rows
    _cfg(spec_tokens=1, spec_verify="packed", router_logits="exact_fp32")
    moe = importlib.import_module("models.demos.motif3.tt.moe")
    use = getattr(getattr(moe, "MotifRouter", None), "_use_logits_fn", None)
    if use is not None:  # the constant must say what tt/moe.py does (D1 changes both together)
        stub = SimpleNamespace(logits_fn=object())
        try:
            got = {M: bool(use(stub, SimpleNamespace(shape=[1, 1, M, 4096]))) for M in (32, 64)}
        except (AttributeError, TypeError) as e:  # MotifRouter's predicate changed shape: D1 re-checks the constant
            print(f"[config] MotifRouter._use_logits_fn not probed with a stub: {e!r}")
        else:
            assert got == {M: M in ROUTER_EXACT_FP32_DECODE_ROWS for M in (32, 64)}, got

    # geometry and value checks
    with pytest.raises(ValueError, match="tile row"):
        _cfg(mesh_shape=(1, 8), spec_tokens=1, spec_verify="auto")  # 2 x 32 lanes on one DP row
    assert _cfg(mesh_shape=(1, 8), spec_tokens=1).wide_rows_per_dp == 0
    for bad in (dict(spec_verify="tall"), dict(wide_step_ratio=0.99), dict(wide_step_ratio=float("nan"))):
        with pytest.raises(ValueError):
            _cfg(**bad)
    with pytest.raises(ValueError):
        _cfg(wide_step_ratio=float("inf"))
    assert _cfg(wide_step_ratio=1).wide_step_ratio == 1.0 and _cfg(wide_step_ratio=1.3).wide_step_ratio == 1.3


# ---- F3N rule R1 (P5_T64_DESIGN.md §2.3, review edit R-E5): every TP-ring collective goes through MotifCCL ----------
_TTNN_COLLECTIVE = re.compile(
    r"(?:^|_)(?:all_gather|all_reduce|reduce_scatter|all_broadcast|all_to_all|point_to_point)(?:_|$)"
)
_TTNN_NORM_STATS = re.compile(r"_(?:pre|post)_all_gather$")  # rms_norm_pre_all_gather & co: norm statistics, not CCLs


def _raw_ttnn_collectives(path: Path):
    """``file:line expr`` of every reference to a ttnn collective op in ``path`` (calls, aliases, ``from ttnn import``,
    ``getattr(ttnn, "...")``), whatever name ``ttnn`` is imported under."""
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = {"ttnn"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {(a.asname or a.name).split(".")[0] for a in node.names if a.name.split(".")[0] == "ttnn"}

    def collective(name: str) -> bool:
        return bool(_TTNN_COLLECTIVE.search(name)) and not _TTNN_NORM_STATS.search(name)

    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "ttnn":
            names = [a.name for a in node.names if collective(a.name)]
            hits += [f"{path.name}:{node.lineno} from {node.module} import {n}" for n in names]
        elif isinstance(node, ast.Attribute) and collective(node.attr):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in roots:
                hits.append(f"{path.name}:{node.lineno} {ast.unparse(node)}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and node.args:
            obj = node.args[0]
            while isinstance(obj, ast.Attribute):
                obj = obj.value
            name = node.args[1] if len(node.args) > 1 else None
            if isinstance(obj, ast.Name) and obj.id in roots and isinstance(name, ast.Constant):
                if isinstance(name.value, str) and collective(name.value):
                    hits.append(f"{path.name}:{node.lineno} {ast.unparse(node)}")
    return hits


def test_r1_no_raw_ttnn_collectives_outside_ccl(tmp_path):
    """F3N R1: the ring-gather race guard (``MotifCCL``, ``ring_gather="safe"``) only covers collectives that go
    through ``tt/ccl.py``. No module under ``tt/`` other than ``ccl.py`` may reference a ttnn collective op directly:
    every new P5 / T64 payload (packed moments up to ``[1,3,2048,32]``, T64's 16-row decode gathers) must reach the
    predicate. ``ccl.py``'s own native calls sit behind ``_ag_race_prone`` (R-E5)."""
    tt_dir = Path(importlib.import_module("models.demos.motif3.tt.model_config").__file__).resolve().parent
    files = sorted(tt_dir.rglob("*.py"))
    assert len(files) > 10 and (tt_dir / "ccl.py") in files
    own = _raw_ttnn_collectives(tt_dir / "ccl.py")
    assert any("ttnn.all_gather" in h for h in own), own  # the scan sees real call sites (no vacuous pass)
    raw = [h for f in files if f.name != "ccl.py" for h in _raw_ttnn_collectives(f)]
    assert not raw, f"raw ttnn collectives outside tt/ccl.py (route them through MotifCCL, F3N rule R1): {raw}"
    # the scan itself: aliases, from-imports, getattr and nested namespaces are caught; norm statistics and strings not
    src = (
        "import ttnn as tn\nimport ttnn\nfrom ttnn import all_gather as ag\nx = tn.experimental.all_gather_async\n"
        "y = getattr(ttnn, 'reduce_scatter')\nz = ttnn.rms_norm_pre_all_gather\ns = 'ttnn.all_gather'\n"
        "w = self_ccl.all_gather\n"
    )
    probe = tmp_path / "probe.py"
    probe.write_text(src)
    hits = _raw_ttnn_collectives(probe)
    assert len(hits) == 3 and all(h.startswith("probe.py:") for h in hits), hits


def test_capture_thread_knob(monkeypatch):
    """E3 (logs/opt/phaseB/B7): ``MOTIF3_CAPTURE_THREAD`` (``main`` default, the release | ``worker``; worker
    lost served 1K / 4K TTFT with the B7 prefill traces), the host thread of every trace capture; case and blanks ignored, anything else refused; ``describe`` shows it."""
    from models.demos.motif3.tt.generator_api import CAPTURE_THREAD_MODES

    assert CAPTURE_THREAD_MODES == ("main", "worker")
    c = _cfg()
    assert c.capture_thread == "main" and "capture_thread=main" in c.describe()
    for v, want in ((" Worker", "worker"), ("main", "main"), (" MAIN", "main"), ("", "main")):
        monkeypatch.setenv("MOTIF3_CAPTURE_THREAD", v)
        assert _cfg().capture_thread == want, v
    monkeypatch.setenv("MOTIF3_CAPTURE_THREAD", "thread")
    with pytest.raises(ValueError, match="MOTIF3_CAPTURE_THREAD"):
        _cfg()
    monkeypatch.delenv("MOTIF3_CAPTURE_THREAD")
    assert "capture_thread=worker" in _cfg(capture_thread="worker").describe()
    with pytest.raises(ValueError, match="capture_thread"):
        _cfg(capture_thread="pool")


def test_prefill_trace_knob(monkeypatch):
    """B7 (docs/OPTIMIZATION_PLAN.md §3.3 B7; logs/opt/phaseB/B7): ``MOTIF3_PREFILL_TRACE`` (``128`` default since its
    gates | ``off`` | ``on`` = ``128`` | a comma list of 128 / 256 / 512, canonical ascending); case and blanks
    ignored, anything else refused; ``describe`` shows it."""
    from models.demos.motif3.tt.generator_api import PREFILL_TRACE_BUCKETS

    assert PREFILL_TRACE_BUCKETS == (128, 256, 512)
    c = _cfg()
    assert c.prefill_trace == "128" and "prefill_trace=128 capture_thread=main" in c.describe()
    for v, want in (("on", "128"), (" 256,128 ", "128,256"), ("OFF", "off"), ("", "128"), ("512", "512")):
        monkeypatch.setenv("MOTIF3_PREFILL_TRACE", v)
        assert _cfg().prefill_trace == want, v
    monkeypatch.setenv("MOTIF3_PREFILL_TRACE", "1024")
    with pytest.raises(ValueError, match="MOTIF3_PREFILL_TRACE"):
        _cfg()
    monkeypatch.delenv("MOTIF3_PREFILL_TRACE")
    assert "prefill_trace=off " in _cfg(prefill_trace="off").describe()
    with pytest.raises(ValueError, match="prefill_trace"):
        _cfg(prefill_trace="2048")
