# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""(d) Real-weight smoke + HF parity: embed + layers 0-2 (global/dense, swa/dense, swa/moe) + final norm + lm_head,
and the MTP head.

A chat-templated prompt longer than the 129-key window runs through the reference prefix model in bf16 and fp32.
Each layer's output is compared with an HF ``MotifDecoderLayer`` that holds the SAME real tensors (built on the meta
device, then ``load_state_dict(assign=True)`` - constructing a decoder layer never runs ``_init_weights``, so
nothing zeroes the loaded parameters). The MTP head is compared with ``hf_reference.HFMTPOracle`` (HF blocks with
the fork's MTP wiring). Skipped when the local shards are missing.
"""

import pytest
import torch

from models.demos.motif3.reference import load_mtp, load_reference_model, reference_to_hf_state_dict
from models.demos.motif3.reference.generate import MotifGenerator
from models.demos.motif3.reference.golden import (
    DEFAULT_PROMPT_MESSAGES,
    TensorRecorder,
    capture_model_goldens,
    export_real_goldens,
    load_golden,
    save_golden,
)
from models.demos.motif3.reference.tokenizer import encode_chat, load_tokenizer

from .common import REAL_LAYERS, max_abs, pcc, real_checkpoint
from .hf_reference import HFMTPOracle, hf_config_from_json, hf_cpu_flash_attention, load_hf_modules

pytestmark = pytest.mark.timeout(3600)


@pytest.fixture(scope="module")
def ckpt():
    return real_checkpoint(REAL_LAYERS)


@pytest.fixture(scope="module")
def tokenizer(ckpt):
    return load_tokenizer(ckpt.dir)


@pytest.fixture(scope="module")
def prompt_ids(tokenizer):
    ids = encode_chat(DEFAULT_PROMPT_MESSAGES, tokenizer)
    assert ids[0] == 1 and ids[-1] == 11  # <|beginoftext|> ... <|assistant|><think>
    assert len(ids) > 129, len(ids)  # exercises the SWA window on layers 1 and 2
    return torch.tensor([ids])


@pytest.fixture(scope="module")
def models(ckpt):
    out = {
        dt: load_reference_model(layer_ids=REAL_LAYERS, dtype=dt, checkpoint=ckpt)
        for dt in (torch.bfloat16, torch.float32)
    }
    yield out
    out.clear()


def test_tokenizer_chat_template(tokenizer):
    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "What is 2+2?"}]
    assert encode_chat(msgs, tokenizer) == [
        1,
        5,
        2,
        201868,
        6411,
        173,
        6,
        5,
        3,
        200555,
        351,
        177,
        170,
        177,
        190,
        6,
        5,
        4,
        11,
    ]
    no_think = encode_chat(msgs, tokenizer, enable_thinking=False)
    assert no_think[-2:] == [11, 12]  # <think></think>
    assert len(tokenizer) == 220160


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_real_prefill_layers_0_2_match_hf(ckpt, models, prompt_ids, dtype):
    ref = models[dtype]
    rec = TensorRecorder(lambda n: n.endswith((".x_in", ".x_out", "router.indices", "router.weights")) or n == "logits")
    with torch.no_grad():
        logits = ref(prompt_ids, tap=rec)
    for name, t in rec.tensors.items():
        assert torch.isfinite(t.float()).all(), name
    assert logits.dtype == torch.float32 and logits.shape == (1, prompt_ids.shape[1], 220160)
    assert rec.tensors["layers.2.moe.router.indices"].shape == (prompt_ids.shape[1], 8)
    torch.testing.assert_close(
        rec.tensors["layers.2.moe.router.weights"].sum(-1), torch.full((prompt_ids.shape[1],), 2.0)
    )

    cfg_mod, mm = load_hf_modules()
    cfg = hf_config_from_json(ckpt.dir, cfg_mod)
    S = prompt_ids.shape[1]
    pos = torch.arange(S)[None]
    sd = ref.state_dict()
    with hf_cpu_flash_attention(), torch.no_grad():
        rotary = mm.MotifRotaryEmbedding(cfg, rope_head_dim=cfg.qk_rope_head_dim)  # global-layer (YaRN) cos/sin
        for i in REAL_LAYERS:
            with torch.device("meta"):
                hf_layer = mm.MotifDecoderLayer(cfg, i)
            p = f"model.layers.{i}."
            layer_sd = reference_to_hf_state_dict({k[len(p) :]: v for k, v in sd.items() if k.startswith(p)})
            hf_layer.load_state_dict(layer_sd, strict=True, assign=True)
            hf_layer.eval()
            x = rec.tensors[f"layers.{i}.x_in"]
            pe = rotary(x[:, :, 0], pos)
            hf_out = hf_layer(
                x,
                attention_mask=None,
                position_ids=pos,
                past_key_value=None,
                use_cache=False,
                cache_position=pos[0],
                position_embeddings=pe,
            )[0]
            ours = rec.tensors[f"layers.{i}.x_out"]
            print(
                f"\n[{dtype}] layer {i} ({ref.args.layer_kind(i)}): exact={torch.equal(hf_out, ours)} "
                f"max|d|={max_abs(hf_out, ours):.3e} pcc={pcc(hf_out, ours):.8f}"
            )
            if dtype == torch.float32:
                torch.testing.assert_close(ours, hf_out, atol=1e-4, rtol=1e-4)
            else:
                assert pcc(ours, hf_out) > 0.99999


def test_real_bf16_vs_fp32_and_absorbed(models, prompt_ids):
    rec = {}
    for dt, m in models.items():
        rec[dt] = TensorRecorder(lambda n: n.endswith(".x_out") or n == "logits")
        with torch.no_grad():
            m(prompt_ids, tap=rec[dt])
    for i in REAL_LAYERS:
        a, b = rec[torch.bfloat16].tensors[f"layers.{i}.x_out"], rec[torch.float32].tensors[f"layers.{i}.x_out"]
        print(f"\nlayer {i} bf16 vs fp32 pcc={pcc(a, b):.6f}")
        assert pcc(a, b) > 0.999
    assert pcc(rec[torch.bfloat16].tensors["logits"], rec[torch.float32].tensors["logits"]) > 0.99
    with torch.no_grad():
        absorbed = models[torch.float32](prompt_ids, attn_mode="absorbed")
    expanded = rec[torch.float32].tensors["logits"]
    assert max_abs(absorbed, expanded) <= 1e-3 * float(expanded.abs().max()), max_abs(absorbed, expanded)


def test_real_decode_matches_prefill(models, prompt_ids):
    model = models[torch.float32]
    S = prompt_ids.shape[1]
    with torch.no_grad():
        full = model(prompt_ids)
    for mode in ("expanded", "absorbed"):
        gen = MotifGenerator(model, 1, S, attn_mode=mode)
        pre = gen.prefill(prompt_ids[:, : S - 4])
        steps = [gen.decode(prompt_ids[:, t]) for t in range(S - 4, S)]
        dec = torch.cat([pre[0], torch.cat(steps, 0)], 0)
        assert max_abs(dec, full[0]) <= 1e-3 * float(full.abs().max()), (mode, max_abs(dec, full[0]))


def test_real_lazy_experts_match_materialized(ckpt, models, prompt_ids):
    lazy = load_reference_model(layer_ids=REAL_LAYERS, dtype=torch.bfloat16, lazy_experts=True, checkpoint=ckpt)
    assert lazy.model.layers["2"].moe.experts.gate_up_proj is None
    with torch.no_grad():
        assert torch.equal(lazy(prompt_ids[:, :64]), models[torch.bfloat16](prompt_ids[:, :64]))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_real_mtp_matches_hf_oracle(ckpt, models, prompt_ids, dtype):
    """Real MTP weights vs the HF-block oracle (fork wiring, ``hf_reference.HFMTPOracle``; tensors read straight from
    the shards). Inputs are the prefix model's post-norm hidden states of the chat prompt and the next-token
    embeddings, S = 144 > 129 keys, so the SWA window, plain RoPE and the 192^-0.5 scale are all exercised."""
    prefix = "model.mtp_layers.0."
    names = ckpt.names_with_prefix(prefix)
    if not names or not all(ckpt.is_local(n) for n in names):
        pytest.skip("MTP shard not local")
    mtp = load_mtp(dtype=dtype, checkpoint=ckpt)
    model = models[dtype]
    cfg_mod, mm = load_hf_modules()
    raw = {n[len(prefix) :]: t for n, t in ckpt.get_many(names, dtype).items()}
    oracle = HFMTPOracle(mm, hf_config_from_json(ckpt.dir, cfg_mod), raw, dtype)
    S = prompt_ids.shape[1] - 1
    pos = torch.arange(S)[None]
    rec = TensorRecorder(lambda n: n in ("final.norm", "input_proj.out"))
    with hf_cpu_flash_attention(), torch.no_grad():
        model(prompt_ids[:, :S], tap=rec)
        h_main = rec.tensors["final.norm"]
        next_emb = model.model.embed_tokens(prompt_ids[:, 1 : S + 1])
        h_in, expect = oracle(h_main, next_emb, pos)
        out = mtp(h_main, next_emb, pos, tap=rec)
        absorbed = mtp(h_main, next_emb, pos, attn_mode="absorbed")
        logits = model.lm_head(out).float()
    print(
        f"\n[{dtype}] real MTP vs HF-block oracle: exact={torch.equal(out, expect)} "
        f"max|d|={max_abs(out, expect):.3e} pcc={pcc(out, expect):.8f}; absorbed pcc={pcc(absorbed, out):.6f}"
    )
    assert out.shape == (1, S, 4096) and torch.isfinite(logits).all()
    assert torch.equal(rec.tensors["input_proj.out"], h_in)
    if dtype == torch.float32:
        torch.testing.assert_close(out, expect, atol=1e-4, rtol=1e-4)
        assert max_abs(absorbed, out) <= 1e-4 * float(out.abs().max())
    else:
        assert pcc(out, expect) > 0.99999
        assert pcc(absorbed, out) > 0.999
    hf_attn = oracle.block.self_attn
    assert mtp.self_attn.window == hf_attn.sliding_window == 129
    assert mtp.self_attn.scale == hf_attn.scaling and abs(mtp.self_attn.scale - 192**-0.5) < 1e-12
    assert not mtp.self_attn.uses_yarn and torch.equal(mtp.self_attn.inv_freq(), hf_attn.swa_rotary_emb.inv_freq)


def test_real_golden_capture(models, prompt_ids, tmp_path):
    model = models[torch.bfloat16]
    keep = lambda n: n.startswith("layers.2.") or n in ("logits", "embed")  # noqa: E731
    g = capture_model_goldens(model, prompt_ids[:, :-2], decode_ids=prompt_ids[:, -2:], include=keep)
    path = save_golden(g, tmp_path / "real_L2.pt")
    g2 = load_golden(path)
    p = g2["prefill"]
    assert p["layers.2.x_in"].dtype == torch.bfloat16 and p["layers.2.x_in"].shape[2:] == (4, 4096)
    assert p["layers.2.self_attn.c_kv"].shape[-1] == 512 and p["layers.2.self_attn.k_pe"].shape[-1] == 64
    assert len(g2["decode"]) == 2 and "layers.2.moe.router.indices" in g2["decode"][0]
    assert g2["meta"]["layers"][2]["window"] == 129 and g2["meta"]["layers"][0]["window"] is None


def test_real_export_goldens_cli_api(tmp_path):
    paths = export_real_goldens(tmp_path, layer_ids=(0, 1, 2), dtype=torch.bfloat16, decode_steps=2)
    assert [p.rsplit("/", 1)[1] for p in paths] == [
        "layer0_bfloat16_expanded.pt",
        "layer1_bfloat16_expanded.pt",
        "layer2_bfloat16_expanded.pt",
        "model_L0-2_bfloat16_expanded.pt",
    ]
    g1, g2 = load_golden(paths[1]), load_golden(paths[2])
    assert g1["meta"]["window"] == 129 and g2["meta"]["is_moe"]
    assert torch.equal(g2["prefill"]["x_in"], g1["prefill"]["x_out"])  # layer 2 input = layer 1 output
    assert "moe.router.scores" in g2["prefill"] and "mlp.act" in g1["prefill"] and len(g2["decode"]) == 2
    assert load_golden(paths[3])["prefill"]["logits"].shape[-1] == 220160
