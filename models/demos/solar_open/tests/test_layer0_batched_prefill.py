# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layer 0 with REAL weights: the packed multi-user prefill form against the per-user form (phase 3a(1)).

A packed pass feeds ``B`` users x ``S`` tokens as ONE ``[1, 1, B*S, H]`` forward with ``batch_size=B``: attention
reshapes to ``[B, 1, S, -1]`` (per-user causal SDPA, per-user paged KV fill), every other block is row-wise and merely
sees ``T = B*S`` rows instead of ``S``. This test isolates where a packed pass departs from ``B`` per-user forwards
(``batch_size=1``, one user at a time -- today's sequential prefill) on the real layer-0 weights and real embeddings:

1. attention alone (post-norm input; packed vs per-user rows), 2. the MoE alone (``T = B*S`` vs ``S`` rows: pure
row-wise numerics), 3. the whole layer; each also against the HF ``SolarOpenDecoderLayer`` reference. Every per-user
comparison must stay at the run-to-run level of the phase-2 tree (the same input through the same ops is bit-identical
on this box; the two forms differ only by matmul blocking at another row count, i.e. bfp8-floor noise).

    pytest models/demos/solar_open/tests/test_layer0_batched_prefill.py -k 1x8
"""

import pytest
import torch
from loguru import logger

import ttnn

from .test_factory import TestFactory, compare_tensors, parametrize_mesh_with_fabric
from .test_layer0_real_weights import (  # noqa: F401 -- layer0_weights is a fixture
    LAYER_IDX,
    REAL_WEIGHT_THRESHOLDS,
    _snapshot_dir,
    _token_ids,
    build_reference_layer,
    layer0_weights,
)
from .unit import test_modules as tm

# Packed-vs-per-user floors. Measured 2026-09-08 (real layer 0, 4 x 128): the row-wise matmuls get ttnn's auto program
# config for T = B x S rows instead of S rows (another bf16 partial-sum order), so nothing is bit-identical: attention
# rows PCC mean 0.99968 / min 0.99852 (max |diff| 0.006), MoE 0.99968 / 0.99937, whole layer 0.99967 / 0.99607 (the low
# rows are the router's near-tie flips at the other accumulation), K/V blocks 0.99992 / 0.99984; both forms equally
# close to HF (attention 0.99959 / 0.99970, MoE 0.99997 / 0.99996, layer 0.99888 / 0.99942). 0.99 on the row minimum
# keeps >= 0.5 of margin to a real defect (a mixed-up user, a wrong RoPE position or page row: PCC < 0.5); the
# HF distance of the packed form must stay within HF_PCC_MARGIN of the per-user form.
PACKED_VS_PER_USER_PCC = 0.99
HF_PCC_MARGIN = 0.002
KV_PCC_MIN = 0.999


def _pcc_rows(a, b):
    """Per-row PCC of two ``[rows, hidden]`` tensors."""
    a = a.float() - a.float().mean(dim=-1, keepdim=True)
    b = b.float() - b.float().mean(dim=-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "batch_size, seq_len, duplicate_users",
    [(4, 128, False), (4, 128, True)],
    ids=["b4_s128", "b4_s128_dup"],
)
@parametrize_mesh_with_fabric([(1, 8)])
def test_layer0_packed_vs_per_user_prefill(
    mesh_device, device_params, batch_size, seq_len, duplicate_users, layer0_weights, reset_seeds
):
    """``duplicate_users``: users 2 and 3 carry user 0's and user 1's tokens -- identical rows at other slots of the
    same pass must stay bit-identical per block (position independence inside a packed pass; the phase-2 sequential
    prefill is trivially position independent). Reported per block, asserted for every block."""
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] != 1 or mesh_shape[1] != 8:
        pytest.skip(f"real-weight layer test is sized for the 1x8 TP=8 mesh, got {mesh_shape}")
    num_tokens = batch_size * seq_len
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    state_dict, embed = layer0_weights
    reference_layer = build_reference_layer(config, state_dict)
    paged_attention_config, page_table_tt = tm.make_paged_attention(mesh_device, batch_size, seq_len)
    decoder_layer = tm.setup_decoder_layer(
        setup, reference_layer, batch_size, seq_len, layer_idx=LAYER_IDX, paged_attention_config=paged_attention_config
    )
    page_table_torch = ttnn.to_torch(ttnn.get_device_tensors(page_table_tt)[0])

    token_ids = _token_ids(_snapshot_dir(), num_tokens, config.vocab_size, setup["model_args"]).reshape(
        batch_size, seq_len
    )
    if duplicate_users:
        half = batch_size // 2
        token_ids = torch.cat([token_ids[:half], token_ids[:half]])  # users half.. repeat users 0..half-1
    hidden_states = embed[token_ids].float()  # [B, S, H] fp32 copies of the bf16 embeddings
    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=1)
    # RoPE for ONE user of S tokens ([1, 1, S, hd] cos/sin): the production packed pass hands the layers exactly
    # these S-row matrices (Model.prepare_inputs_prefill), the per-user pass too.
    _, position_embeddings_ref, rope_mats, _ = tm.build_rope_inputs(
        setup, config, hidden_states[:1], 1, seq_len, 0, 1, False
    )
    replicate = ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_shape)

    def upload(x):  # [rows, H] fp32 -> replicated bf16 [1, 1, rows, H]
        return ttnn.from_torch(
            x.reshape(1, 1, -1, config.hidden_size),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )

    def download(t, rows):
        return ttnn.to_torch(t, mesh_composer=composer)[..., :rows, : config.hidden_size].reshape(rows, -1).float()

    def per_user_page_table(u):
        return ttnn.from_torch(
            page_table_torch[u : u + 1],
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    problems = []

    def check_duplicates(name, packed):
        """Rows of the duplicated users must equal their originals bit for bit (same pass, other slot)."""
        if not duplicate_users:
            return
        half = batch_size // 2
        rows = packed.reshape(batch_size, seq_len, -1)
        for u in range(half):
            a, b = rows[u], rows[u + half]
            same = torch.equal(a, b)
            pcc = float(_pcc_rows(a, b).min())
            logger.info(
                f"[{name}] user {u + half} (copy of user {u}, other slot): bit-identical {same}, row PCC min {pcc:.6f}"
            )
            if not same:
                problems.append(
                    f"{name}: user {u + half} is not bit-identical to its copy user {u} (row PCC min {pcc:.6f})"
                )

    def compare(name, packed, per_user):
        pcc = _pcc_rows(packed, per_user)
        per_user_pcc = pcc.reshape(batch_size, seq_len).mean(dim=1)
        diff = (packed - per_user).abs()
        identical = int((diff.reshape(batch_size, seq_len, -1).amax(dim=(1, 2)) == 0).sum())
        logger.info(
            f"[{name}] packed vs per-user: row PCC min {pcc.min():.6f} mean {pcc.mean():.6f}; per user "
            f"{[round(float(p), 6) for p in per_user_pcc]}; max |diff| {diff.max():.4f}; users bit-identical {identical}/{batch_size}"
        )
        if pcc.min() < PACKED_VS_PER_USER_PCC:
            problems.append(
                f"{name}: packed-vs-per-user row PCC min {pcc.min():.5f} (per user {per_user_pcc.tolist()})"
            )

    # ---- 1. attention alone on the post-input-norm rows (HF norm on the fp32 copy, then bf16 on device) ----
    with torch.no_grad():
        attn_in = reference_layer.input_layernorm(hidden_states)  # [B, S, H]
        attn_ref, _ = reference_layer.self_attn(
            hidden_states=attn_in, position_embeddings=position_embeddings_ref, attention_mask=mask
        )
    attn_in_bf16 = attn_in.to(torch.bfloat16).float()
    tt_attn_packed = decoder_layer.self_attn(
        upload(attn_in_bf16.reshape(num_tokens, -1)),
        rope_mats=rope_mats,
        position_idx=None,
        page_table=page_table_tt,
        is_decode=False,
        user_id=0,
        batch_size=batch_size,
    )
    attn_packed = download(tt_attn_packed, num_tokens)
    tt_attn_packed.deallocate(True)
    attn_per_user = []
    for u in range(batch_size):
        tt_out = decoder_layer.self_attn(
            upload(attn_in_bf16[u]),
            rope_mats=rope_mats,
            position_idx=None,
            page_table=per_user_page_table(u),
            is_decode=False,
            user_id=0,
            batch_size=1,
        )
        attn_per_user.append(download(tt_out, seq_len))
        tt_out.deallocate(True)
    attn_per_user = torch.cat(attn_per_user)
    check_duplicates("attention", attn_packed)
    compare("attention", attn_packed, attn_per_user)
    _, pcc_p = compare_tensors(attn_packed, attn_ref.reshape(num_tokens, -1), mesh_device, 0.9)
    _, pcc_u = compare_tensors(attn_per_user, attn_ref.reshape(num_tokens, -1), mesh_device, 0.9)
    logger.info(f"[attention] vs HF: packed {pcc_p}, per-user {pcc_u}")

    # ---- 2. MoE alone: T = B*S rows vs S rows (row-wise ops; the only difference is the row count) ----
    with torch.no_grad():
        moe_in = reference_layer.post_attention_layernorm(hidden_states + attn_ref)
        moe_ref = reference_layer.mlp(moe_in).reshape(num_tokens, -1)
    moe_in_bf16 = moe_in.to(torch.bfloat16).float()
    tt_moe = decoder_layer.mlp(upload(moe_in_bf16.reshape(num_tokens, -1)), is_decode=False)
    moe_packed = download(tt_moe, num_tokens)
    tt_moe.deallocate(True)
    moe_per_user = []
    for u in range(batch_size):
        tt_out = decoder_layer.mlp(upload(moe_in_bf16[u]), is_decode=False)
        moe_per_user.append(download(tt_out, seq_len))
        tt_out.deallocate(True)
    moe_per_user = torch.cat(moe_per_user)
    check_duplicates("moe", moe_packed)
    compare("moe", moe_packed, moe_per_user)
    _, pcc_p = compare_tensors(moe_packed, moe_ref, mesh_device, 0.9)
    _, pcc_u = compare_tensors(moe_per_user, moe_ref, mesh_device, 0.9)
    logger.info(f"[moe] vs HF: packed {pcc_p}, per-user {pcc_u}")

    # ---- 3. whole layer ----
    with torch.no_grad():
        layer_ref = reference_layer(hidden_states, attention_mask=mask, position_embeddings=position_embeddings_ref)
        layer_ref = layer_ref.reshape(num_tokens, -1)
    tt_layer = decoder_layer(
        upload(hidden_states.reshape(num_tokens, -1)),
        position_embeddings=rope_mats,
        position_idx=None,
        page_table=page_table_tt,
        is_decode=False,
        user_id=0,
        batch_size=batch_size,
    )
    layer_packed = download(tt_layer, num_tokens)
    tt_layer.deallocate(True)
    # K/V blocks the packed pass wrote (per-user page-table rows, batch_idx 0 per user), device 0
    k_cache, v_cache = decoder_layer.self_attn.layer_past
    k_packed = ttnn.to_torch(ttnn.get_device_tensors(k_cache)[0]).float().clone()
    v_packed = ttnn.to_torch(ttnn.get_device_tensors(v_cache)[0]).float().clone()
    ttnn.mul(k_cache, 0, output_tensor=k_cache)
    ttnn.mul(v_cache, 0, output_tensor=v_cache)
    layer_per_user = []
    for u in range(batch_size):
        tt_out = decoder_layer(
            upload(hidden_states[u]),
            position_embeddings=rope_mats,
            position_idx=None,
            page_table=per_user_page_table(u),
            is_decode=False,
            user_id=0,
            batch_size=1,
        )
        layer_per_user.append(download(tt_out, seq_len))
        tt_out.deallocate(True)
    layer_per_user = torch.cat(layer_per_user)
    check_duplicates("decoder", layer_packed)
    compare("decoder", layer_packed, layer_per_user)
    passing_p, pcc_p = compare_tensors(layer_packed, layer_ref, mesh_device, REAL_WEIGHT_THRESHOLDS["decoder"])
    passing_u, pcc_u = compare_tensors(layer_per_user, layer_ref, mesh_device, REAL_WEIGHT_THRESHOLDS["decoder"])
    logger.info(f"[decoder] vs HF: packed {pcc_p}, per-user {pcc_u}")
    if not passing_p:
        problems.append(f"packed decoder layer vs HF: {pcc_p}")
    if not passing_u:
        problems.append(f"per-user decoder layer vs HF: {pcc_u}")
    hf_p, hf_u = float(str(pcc_p).split()[-1]), float(str(pcc_u).split()[-1])
    if hf_p < hf_u - HF_PCC_MARGIN:
        problems.append(f"packed decoder layer is farther from HF than the per-user form: {hf_p:.5f} vs {hf_u:.5f}")

    # ---- 4. the KV blocks written by the packed pass equal the per-user fills ----
    k_seq = ttnn.to_torch(ttnn.get_device_tensors(k_cache)[0]).float()
    v_seq = ttnn.to_torch(ttnn.get_device_tensors(v_cache)[0]).float()
    for name, a, b in (("k", k_packed, k_seq), ("v", v_packed, v_seq)):
        pcc = float(_pcc_rows(a.reshape(1, -1), b.reshape(1, -1))[0])
        logger.info(f"[kv {name}] packed fill vs per-user fills: bit-identical {torch.equal(a, b)}, PCC {pcc:.6f}")
        if pcc < KV_PCC_MIN:
            problems.append(f"{name} cache of the packed pass differs from the per-user fills (PCC {pcc:.5f})")

    assert not problems, "packed prefill differs from the per-user form:\n" + "\n".join(problems)
