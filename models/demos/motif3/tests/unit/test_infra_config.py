# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for tt/model_config.py (no device).

Run device-hidden (the root conftest opens the UMD cluster even for collection), e.g. with the wrapper::

    scripts/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib -q \
        models/demos/motif3/tests/unit/test_infra_config.py
"""

import dataclasses
import importlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.motif3.tt import generator_api as api
from models.demos.motif3.tt.model_config import (
    CACHE_FORMAT_VERSION,
    COMPUTE_ROLES,
    DEFAULT_HF_META_DIR,
    DEFAULT_L1_SMALL_SIZE,
    DEFAULT_WEIGHTS_DIR,
    FP32_ACC_OFF_ROLES,
    MeshAxes,
    MotifTTConfig,
    compute_config_descriptor,
    device_params,
    experts_down_pc,
    experts_gate_up_pc,
    flash_mla_decode_pc,
    make_compute_kernel_config,
    mcast1d_matmul_pc,
    mesh_l1_small_bytes,
    mesh_shape_from_env,
    require_l1_small,
    resolve_weights_dir,
    rope_scaling_of,
    sdpa_prefill_chunks,
    sdpa_prefill_pc,
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
