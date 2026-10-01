# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for tt/model_config.py (no device).

Run device-hidden (the root conftest opens the UMD cluster even for collection):

    unshare -Urm --propagation private bash -c 'mount -t tmpfs none /dev/tenstorrent && \
      cd tt-metal && source python_env/bin/activate && \
      python -m pytest --noconftest -o addopts="" --import-mode=importlib -q \
        models/demos/motif3/tests/unit/test_infra_config.py'
"""

import math

import pytest

import ttnn
from models.demos.motif3.tt.model_config import (
    CACHE_FORMAT_VERSION,
    DEFAULT_HF_META_DIR,
    MeshAxes,
    MotifTTConfig,
    device_params,
)

HF_META = str(DEFAULT_HF_META_DIR)


def _cfg(**kw):
    return MotifTTConfig.from_hf_config(HF_META, **kw)


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
            assert L.window is None and L.rope_kind == "yarn"
            assert abs(L.softmax_scale - 0.14467963) < 1e-8
        else:
            assert L.window == 129 and L.sliding_window_size == 129 and L.rope_kind == "plain"
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


def test_kv_pool_and_buckets():
    cfg = _cfg()
    assert cfg.kv_block_size == 64 and cfg.kv_pool_tokens == 262144 and cfg.max_batch == 32
    # vllm_tt_plugin worker.get_num_available_blocks_tt: ceil((pool + block * max_batch) / block)
    assert cfg.kv_num_blocks == math.ceil((262144 + 64 * 32) / 64) == 4128
    assert cfg.kv_pool_tokens_allocated == 264192
    assert cfg.kv_blocks_per_seq == 512 and cfg.kv_cache_shape == (4128, 1, 64, 576)
    assert cfg.kv_cache_bytes_per_chip() == pytest.approx(8.57e9, rel=2e-3)  # design §1.1
    assert cfg.prefill_buckets == (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    assert [cfg.prefill_bucket(n) for n in (1, 128, 129, 4000, 32768)] == [128, 128, 256, 4096, 32768]
    with pytest.raises(ValueError):
        cfg.prefill_bucket(32769)


def test_dtypes_compute_and_cache_paths(tmp_path, monkeypatch):
    cfg = _cfg()
    d = cfg.dtypes
    assert d.routed_experts == ttnn.bfloat8_b and d.attention == ttnn.bfloat16 and d.kv_cache == ttnn.bfloat8_b
    assert d.router == ttnn.bfloat16 and d.mhc == ttnn.bfloat16 and d.lm_head == ttnn.bfloat16
    assert d.embedding == ttnn.bfloat16 and d.router_bias == ttnn.float32 and d.activations == ttnn.bfloat16
    assert d.tag == "e8s8d8a16r16m16l16v16"
    ck = cfg.compute_config("router")
    assert ck.math_fidelity == ttnn.MathFidelity.HiFi4 and ck.fp32_dest_acc_en and not ck.math_approx_mode
    ce = cfg.compute_config("experts")
    assert ce.math_fidelity == ttnn.MathFidelity.HiFi2 and ce.fp32_dest_acc_en
    assert cfg.compute_config("router") is ck  # cached
    with pytest.raises(KeyError):
        cfg.compute_config("nope")

    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("MOTIF3_NUM_LAYERS", "3")
    monkeypatch.setenv("MOTIF3_KV_POOL_TOKENS", "131072")
    c2 = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(8, 4))
    assert c2.num_layers == 3 and len(c2.layers) == 3 and c2.num_hidden_layers == 53
    assert c2.kv_num_blocks == (131072 + 2048) // 64
    tag = f"motif3-2ed2ed5c-c{CACHE_FORMAT_VERSION}-e8s8d8a16r16m16l16v16"
    assert c2.cache_version_tag == tag
    assert c2.cache_dir == tmp_path / tag / "mesh8x4"
    assert c2.cache_file("attn.wq_b", 2) == tmp_path / tag / "mesh8x4" / "L02" / "attn.wq_b"
    assert c2.cache_file("lm_head", None) == tmp_path / tag / "mesh8x4" / "global" / "lm_head"


def test_device_params(monkeypatch):
    p = device_params()
    assert p["fabric_config"] == ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    assert p["trace_region_size"] == 268435456
    monkeypatch.setenv("MOTIF3_FABRIC", "FABRIC_1D_RING")
    assert device_params()["fabric_config"] == ttnn.FabricConfig.FABRIC_1D_RING
    assert device_params(trace_region_size=1 << 20, l1_small_size=16384) == {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "trace_region_size": 1 << 20,
        "l1_small_size": 16384,
    }
    with pytest.raises(ValueError):
        device_params("FABRIC_NOPE")


def test_from_dict_and_validation():
    import json

    d = json.load(open(f"{HF_META}/config.json"))
    cfg = MotifTTConfig.from_hf_config(d, mesh_shape=(4, 8))
    assert cfg.n_heads == 80 and cfg.eos_token_ids == (0,)  # no generation_config next to a dict
    with pytest.raises(ValueError):
        MotifTTConfig.from_hf_config(d, mesh_shape=(4, 8), num_experts=100)
    with pytest.raises(ValueError):
        MotifTTConfig.from_hf_config(d, mesh_shape=(4, 8), num_layers=60)
