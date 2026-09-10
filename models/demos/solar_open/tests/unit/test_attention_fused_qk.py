# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device gate of the phase-3e (lane B1) fused decode attention chain: the fused arm must be BIT-IDENTICAL to the
legacy chain, per user and on every device replica.

For B in {1, 8, 16, 32} on the 1x8 mesh (random weights, the test_modules machinery: a random 64-token prefix per
user written through the unchanged prefill path, then one decode step at slot 64) the legacy arm (``fused_qk=False``:
``nlp_create_qkv_heads_decode`` on B cores, ``rotary_embedding_llama`` on Q and K, two ``paged_update_cache``) and the
fused arm (``fused_qk=True``: create_heads on the 2B-core grid with ``overlap_qk_coregrid=False``,
``rotary_embedding_llama_fused_qk``, ``paged_fused_update_cache``; RotarySetup built with ``use_qk_fused=True``) run
on separate paged KV caches from the same inputs. ``torch.equal`` is required on: the rotated Q handed to the SDPA
(``decode_qkv_heads``), every K / V page written, and the attention output after o_proj + all-reduce -- per device
on all 8 devices (Q and the caches hold each device's own TP shard of the heads; the output is replicated and is also
checked for replica identity; a placement mistake makes a core read another user's row silently, without a TT_FATAL). The
o_proj arm (``decode_out_cores=(8, 8)`` from the interleaved in0, PCC-gated, not bit-identical) is run once per batch
and its PCC against the legacy output logged and floored loosely; its fp32-reference PCC lives in
``tests/perf/test_config_candidates.py`` (``o_proj_8x8_interleaved``).

    pytest models/demos/solar_open/tests/unit/test_attention_fused_qk.py -k 1x8 -x -p no:cacheprovider
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.solar_open.tests.unit.test_modules import (
    ACTIVATION_DTYPE,
    assert_replicated,
    build_decode_context,
    make_paged_attention,
    setup_decoder_layer,
    setup_reference_layer,
)
from models.demos.solar_open.tt.attention.decode import decode_qkv_heads
from models.demos.solar_open.tt.attention.kv_cache import init_kv_cache
from models.demos.solar_open.tt.attention_configs import SolarOpenAttentionProgramConfig
from models.demos.solar_open.tt.model import create_rope_setup

OPROJ_PCC_FLOOR = 0.998  # loose: the strict gate is the fp32-reference PCC of test_config_candidates + teacher-forced


def _per_device(tt_tensor):
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tt_tensor)]


def _assert_equal_per_device(a, b, label):
    assert len(a) == len(b)
    for d, (x, y) in enumerate(zip(a, b)):
        assert x.shape == y.shape, f"{label}: device {d} shapes {tuple(x.shape)} vs {tuple(y.shape)}"
        if not torch.equal(x, y):
            diff = (x.float() - y.float()).abs()
            rows = int((diff.reshape(-1, diff.shape[-1]).amax(dim=-1) > 0).sum())
            raise AssertionError(
                f"{label}: device {d} differs between the legacy and the fused chain (max |diff| {diff.max().item():.4g}, "
                f"{rows} rows) -- the fused arm must be bit-identical"
            )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("batch_size", [1, 8, 16, 32], ids=["b1", "b8", "b16", "b32"])
@parametrize_mesh_with_fabric([(1, 8)])
def test_fused_qk_matches_legacy(mesh_device, device_params, batch_size, reset_seeds):
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    hidden_size = config.hidden_size
    paged_attention_config, page_table = make_paged_attention(mesh_device, batch_size, 1)
    reference_layer = setup_reference_layer(setup, layer_idx=0)
    decoder_layer = setup_decoder_layer(
        setup, reference_layer, batch_size, 1, layer_idx=0, paged_attention_config=paged_attention_config
    )
    attn = decoder_layer.self_attn
    context = build_decode_context(setup, config, batch_size, 0, batch_size)
    context_len = context.context_len

    # the decode step: bf16-exact hidden states, RoPE / cache position = the slot after the prefix
    hidden = torch.randn(1, 1, batch_size, hidden_size).to(torch.bfloat16).float()
    tt_hidden = ttnn.from_torch(hidden, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ACTIVATION_DTYPE)
    positions = torch.full((batch_size,), context_len, dtype=torch.int64)
    tt_position_idx = ttnn.from_torch(
        positions.to(torch.int32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
    )
    num_local_heads = attn.mesh_config.shard_size(attn.config.num_heads)
    num_local_kv_heads = attn.mesh_config.shard_size(attn.config.num_kv_heads)
    qkv_n = (num_local_heads + 2 * num_local_kv_heads) * attn.config.head_dim

    def fill_prefix(cache):
        for user in range(batch_size):
            tt_prefix = ttnn.from_torch(
                context.prefix[user].reshape(1, 1, context_len, hidden_size),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ACTIVATION_DTYPE,
            )
            out = attn(
                tt_prefix,
                rope_mats=context.prefix_rope_mats,
                position_idx=None,
                page_table=page_table,
                kv_cache=cache,
                is_decode=False,
                user_id=user,
            )
            out.deallocate(True)
            tt_prefix.deallocate(True)

    def run_arm(program_config, rope_setup, label):
        cache = init_kv_cache(mesh_device, attn.config, attn.mesh_config, paged_attention_config)
        fill_prefix(cache)
        attn.program_config = program_config
        attn.transformation_mats = rope_setup.get_both_trans_mats()
        rope_mats = rope_setup.get_rot_mats(positions)  # get_rot_idxs repeats the positions for the K half when fused
        expected_rows = 2 * batch_size if rope_setup.use_qk_fused else batch_size
        assert rope_mats[0].shape[1] == expected_rows, f"{label}: cos rows {rope_mats[0].shape[1]} != {expected_rows}"
        out = attn(
            tt_hidden,
            rope_mats=rope_mats,
            position_idx=tt_position_idx,
            page_table=page_table,
            kv_cache=cache,
            is_decode=True,
        )
        assert_replicated(out, f"attention output ({label})")
        out_t = _per_device(out)
        out.deallocate(True)
        # the rotated Q handed to the SDPA (the KV write of this second call is idempotent: same rows, same values)
        xqkv = ttnn.matmul(
            tt_hidden,
            attn.weights.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=program_config.get_decode_qkv_config(batch_size, qkv_n, hidden_size),
            compute_kernel_config=program_config.get_decode_qkv_compute_config(mesh_device.arch()),
        )
        tt_q = decode_qkv_heads(
            xqkv,
            rope_mats,
            cache,
            attn.config,
            attn.mesh_config,
            mesh_device,
            program_config,
            attn.transformation_mats["decode"],
            attn.kv_mem_cfg,
            tt_position_idx,
            page_table,
            batch_size,
        )
        # Q and the K / V caches are TP-sharded (each device holds its own heads), so they are compared per device
        # between the arms below and NOT checked for replica identity (only the all-reduced output is replicated).
        q_t = _per_device(tt_q)
        tt_q.deallocate(True)
        k_t, v_t = _per_device(cache[0]), _per_device(cache[1])
        logger.info(f"{label}: output {tuple(out_t[0].shape)}, Q {tuple(q_t[0].shape)}, K pages {tuple(k_t[0].shape)}")
        return out_t, q_t, k_t, v_t

    legacy_rope = create_rope_setup(mesh_device, config, max_local_batch_size=batch_size, use_qk_fused=False)
    fused_rope = create_rope_setup(mesh_device, config, max_local_batch_size=batch_size, use_qk_fused=True)
    legacy = run_arm(SolarOpenAttentionProgramConfig(fused_qk=False, decode_out_cores=None), legacy_rope, "legacy")
    fused = run_arm(SolarOpenAttentionProgramConfig(fused_qk=True, decode_out_cores=None), fused_rope, "fused_qk")

    _assert_equal_per_device(legacy[1], fused[1], "rotated Q")
    _assert_equal_per_device(legacy[2], fused[2], "K cache pages")
    _assert_equal_per_device(legacy[3], fused[3], "V cache pages")
    _assert_equal_per_device(legacy[0], fused[0], "attention output")
    logger.info(f"b{batch_size}: fused_qk chain bit-identical to the legacy chain (Q, K / V pages, output; 8 replicas)")

    # o_proj lever (slice 2): not bit-identical (K-block order); loose mutual PCC here, fp32-reference PCC in
    # tests/perf/test_config_candidates.py and the teacher-forced floors on the model
    oproj = run_arm(SolarOpenAttentionProgramConfig(fused_qk=False, decode_out_cores=(8, 8)), legacy_rope, "o_proj_8x8")
    _assert_equal_per_device(legacy[1], oproj[1], "rotated Q (o_proj arm)")
    _assert_equal_per_device(legacy[2], oproj[2], "K cache pages (o_proj arm)")
    _, pcc = comp_pcc(legacy[0][0].float(), oproj[0][0].float(), 0.0)
    pcc = float(pcc)
    max_diff = (legacy[0][0].float() - oproj[0][0].float()).abs().max().item()
    logger.info(f"b{batch_size}: o_proj (8,8) interleaved-in0 vs auto: PCC {pcc:.6f}, max |diff| {max_diff:.4g}")
    assert pcc >= OPROJ_PCC_FLOOR, f"o_proj (8,8) arm PCC {pcc} vs the auto o_proj below {OPROJ_PCC_FLOOR}"
