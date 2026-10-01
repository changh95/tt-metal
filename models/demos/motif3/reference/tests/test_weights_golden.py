# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""weights.py (index-driven lazy loading, name mapping, clear errors) and golden.py (capture / save / load)."""

import json
import shutil

import pytest
import torch

from models.demos.motif3.reference import (
    MissingWeightsError,
    MotifCheckpoint,
    build_random_model,
    hf_to_reference_state_dict,
    load_reference_model,
    random_state_dict,
    reference_to_hf_state_dict,
    tiny_random_args,
)
from models.demos.motif3.reference.golden import (
    capture_layer_goldens,
    capture_model_goldens,
    load_golden,
    random_streams,
    save_golden,
)

from .common import max_abs, rand_ids
from .hf_reference import HF_META_DIR

INDEX = HF_META_DIR / "model.safetensors.index.json"


def _write_tiny_checkpoint(path, args, sd_ref, n_shards=3):
    """Save reference weights as an HF-named, sharded safetensors checkpoint with an index + config.json."""
    from safetensors.torch import save_file

    hf_sd = reference_to_hf_state_dict(sd_ref)
    names = sorted(hf_sd)
    weight_map, shards = {}, [dict() for _ in range(n_shards)]
    for i, n in enumerate(names):
        shards[i % n_shards][n] = hf_sd[n].to(torch.bfloat16).contiguous()
    for s, tensors in enumerate(shards):
        fn = f"model-{s + 1:05d}-of-{n_shards:05d}.safetensors"
        save_file(tensors, str(path / fn))
        weight_map.update({n: fn for n in tensors})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    cfg = args.to_hf_config_dict()
    cfg["eos_token_id"] = list(args.eos_token_ids)
    (path / "config.json").write_text(json.dumps(cfg))
    return weight_map


def test_name_mapping_roundtrip():
    args = tiny_random_args()
    sd = random_state_dict(args, seed=0)
    hf_sd = reference_to_hf_state_dict(sd)
    assert (
        "model.layers.0.mhc_attn.proj_pre.weight" in hf_sd and "model.layers.0.mhc_attn.proj_merged.weight" not in hf_sd
    )
    assert hf_sd["model.layers.3.mhc_ffn.proj_res.weight"].shape == (16, 4 * args.hidden_size)
    back = hf_to_reference_state_dict(hf_sd)
    assert back.keys() == sd.keys() and all(torch.equal(back[k], sd[k]) for k in sd)
    # reference parameter names == checkpoint names for everything except the merged mHC projection
    real = set(json.loads(INDEX.read_text())["weight_map"])
    for i in (0, 2):
        layer = {k for k in hf_sd if k.startswith(f"model.layers.{i}.")}
        assert layer == {k for k in real if k.startswith(f"model.layers.{i}.")}, i


def test_tiny_checkpoint_roundtrip_and_lazy_experts(tmp_path):
    args = tiny_random_args()
    sd = random_state_dict(args, seed=3)
    _write_tiny_checkpoint(tmp_path, args, sd)
    ckpt = MotifCheckpoint(tmp_path)
    assert ckpt.args() == args.replace()  # config.json round trip
    ids = rand_ids(args, 1, 40, seed=1)
    full = load_reference_model(tmp_path, layer_ids=None, dtype=torch.bfloat16, checkpoint=ckpt)
    lazy = load_reference_model(tmp_path, layer_ids=None, dtype=torch.bfloat16, lazy_experts=True, checkpoint=ckpt)
    assert lazy.model.layers["2"].moe.experts.gate_up_proj is None
    direct = build_random_model(
        args, dtype=torch.bfloat16, state_dict={k: v.to(torch.bfloat16).float() for k, v in sd.items()}
    )
    with torch.no_grad():
        a, b, c = full(ids), lazy(ids), direct(ids)
    assert torch.equal(a, b) and torch.equal(a, c)
    prefix = load_reference_model(tmp_path, layer_ids=range(3), dtype=torch.float32, checkpoint=ckpt)
    assert prefix.model.layer_ids == [0, 1, 2] and prefix.dtype == torch.float32


def test_missing_and_incomplete_shards_raise_clear_errors(tmp_path):
    args = tiny_random_args(num_hidden_layers=3)
    weight_map = _write_tiny_checkpoint(tmp_path, args, random_state_dict(args, seed=4))
    victim = "model.layers.1.self_attn.wq_a.weight"
    shard = tmp_path / weight_map[victim]
    data = shard.read_bytes()
    shard.write_bytes(data[: len(data) // 2])  # simulate a shard still being downloaded
    ckpt = MotifCheckpoint(tmp_path)
    assert not ckpt.is_local(victim)
    with pytest.raises(MissingWeightsError, match="incomplete"):
        ckpt.get(victim)
    shard.unlink()
    with pytest.raises(MissingWeightsError, match=f"lives in shard {weight_map[victim]}.*not present"):
        ckpt.get(victim)
    with pytest.raises(MissingWeightsError, match="not a tensor"):
        ckpt.get("model.layers.99.nope")
    with pytest.raises(MissingWeightsError):
        load_reference_model(tmp_path, layer_ids=(0, 1), checkpoint=ckpt)
    # the real index: a layer whose shards are absent fails with the shard name
    real_dir = tmp_path / "real_index_only"
    real_dir.mkdir()
    shutil.copy(INDEX, real_dir / INDEX.name)
    shutil.copy(HF_META_DIR / "config.json", real_dir / "config.json")
    real = MotifCheckpoint(real_dir)
    with pytest.raises(MissingWeightsError, match=r"model-00\d{3}-of-00155\.safetensors"):
        real.get("model.layers.40.self_attn.wq_a.weight")
    assert real.local_layers() == []


@pytest.mark.parametrize("layer_idx", [1, 2], ids=["swa_dense", "swa_moe"])
def test_capture_layer_goldens_roundtrip(tmp_path, layer_idx):
    args = tiny_random_args()
    model = build_random_model(args, seed=6)
    layer = model.model.layers[str(layer_idx)]
    x = random_streams(args, 1, 140, seed=1, dtype=torch.float32, embed_weight=model.model.embed_tokens.weight)
    g = capture_layer_goldens(layer, x, decode_steps=3)
    path = save_golden(g, tmp_path / f"layer{layer_idx}.pt")
    g2 = load_golden(path)  # weights_only=True loadable
    assert g2["meta"]["window"] == 129 and g2["meta"]["decode_steps"] == 3 and len(g2["decode"]) == 3
    p = g2["prefill"]
    for k in ("x_in", "mhc_attn.h_pre", "mhc_attn.h_res", "self_attn.q_pe", "self_attn.c_kv", "self_attn.out", "x_out"):
        assert k in p, k
    if layer_idx == 2:
        assert p["moe.router.indices"].shape == (137, 8) and "moe.shared_experts.act" in p and "moe.out" in p
    else:
        assert "mlp.act" in p
    full = g2["full"]["x_out"]
    torch.testing.assert_close(p["x_out"], full[:, :137], atol=1e-5, rtol=1e-5)
    for s, step in enumerate(g2["decode"]):
        assert step["positions"].item() == 137 + s
        torch.testing.assert_close(step["x_out"], full[:, 137 + s : 138 + s], atol=1e-5, rtol=1e-5)
    assert g2["cache"]["c_kv"].shape == (1, 140, args.kv_lora_rank)


def test_capture_model_goldens(tmp_path):
    args = tiny_random_args()
    model = build_random_model(args, layer_ids=range(3), seed=7, dtype=torch.bfloat16)
    ids = rand_ids(args, 1, 30, seed=2)
    g = capture_model_goldens(model, ids[:, :26], decode_ids=ids[:, 26:])
    assert g["meta"]["dtype"] == "bfloat16" and [m["kind"] for m in g["meta"]["layers"]][2] == "swa/moe"
    assert torch.equal(g["prefill"]["layers.1.x_in"], g["prefill"]["layers.0.x_out"])
    with torch.no_grad():
        full = model(ids)
    # bf16: cached vs cache-free attention may differ by an ulp of the bf16 attention output
    assert max_abs(g["prefill"]["logits"], full[:, :26]) < 0.1
    for t, step in enumerate(g["decode"]):
        assert max_abs(step["logits"][:, 0], full[:, 26 + t]) < 0.1
    save_golden(g, tmp_path / "m.pt")
    assert load_golden(tmp_path / "m.pt")["prefill"]["logits"].shape == (1, 26, args.vocab_size)


def test_capture_expert_goldens_reconstructs_moe():
    from models.demos.motif3.reference.golden import capture_expert_goldens

    args = tiny_random_args()
    model = build_random_model(args, seed=8)
    moe = model.model.layers["2"].moe
    x = torch.randn(1, 12, args.hidden_size, generator=torch.Generator().manual_seed(3))
    g = capture_expert_goldens(moe, x)
    total = torch.zeros(12, args.hidden_size)
    for e in sorted(g["experts"]):
        d = g["experts"][e]
        assert (
            d["gate"].shape == (len(d["tokens"]), args.moe_intermediate_size) and d["out"].shape[-1] == args.hidden_size
        )
        total.index_add_(0, d["tokens"], d["out"].float() * d["weights"][:, None])
    with torch.no_grad():
        shared = moe.shared_experts(x.reshape(-1, args.hidden_size))
        ref = moe(x)
    torch.testing.assert_close((total + shared.float()).view(1, 12, -1).to(x.dtype), ref, atol=1e-6, rtol=1e-6)
    assert torch.equal(g["router"]["indices"].flatten().unique(), torch.tensor(sorted(g["experts"])))
