# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP head (``model.mtp_layers.0``) vs an oracle wired from HF blocks, on a tiny config.

HF ships no MTP module. ``hf_reference.HFMTPOracle`` builds HF ``MotifDecoderLayer`` with the fork's
``_make_mtp_hf_config`` overrides (dense FFN, no mHC, SWA on every layer) plus two ``MotifRMSNorm`` and wires them
like the fork's ``MotifMultiTokenPredictorLayer.forward`` (motif_mtp.py:160-170), which is the same as training's
``_run_mtp_block`` (model.py:865-876). ``test_real_weights.py`` repeats the comparison with the real MTP tensors.
``swa_rope_theta`` differs from the global theta here, so taking the wrong RoPE table cannot go unnoticed.
"""

import pytest
import torch

from models.demos.motif3.reference import LatentKVCache, MotifMTP, tiny_random_args
from models.demos.motif3.reference.golden import TensorRecorder
from models.demos.motif3.reference.weights import random_state_dict, round_state_dict

from .common import max_abs
from .hf_reference import HFMTPOracle, hf_config_from_args, hf_cpu_flash_attention, load_hf_modules

B, S = 2, 150  # S > the 129-key window


def _mtp_and_oracle(dtype: torch.dtype, seed: int = 3):
    args = tiny_random_args(swa_rope_theta=5000.0)
    sd = random_state_dict(args, layer_ids=[], seed=seed, include_mtp=True)
    if dtype != torch.float32:
        sd = round_state_dict(sd, dtype)
    sd = {k[len("mtp.") :]: v for k, v in sd.items() if k.startswith("mtp.")}
    mtp = MotifMTP(args).to(dtype)
    mtp.load_state_dict(sd, strict=True)
    cfg_mod, mm = load_hf_modules()
    return args, mtp, HFMTPOracle(mm, hf_config_from_args(args, cfg_mod), sd, dtype)


def _inputs(args, dtype: torch.dtype, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    h_main = torch.randn(B, S, args.hidden_size, generator=g).to(dtype)  # stands in for the main post-norm hidden
    e_next = (3 * torch.randn(B, S, args.hidden_size, generator=g)).to(dtype)  # embed(t+1); RMS 3, so embed_norm acts
    return h_main, e_next, torch.arange(S)[None].expand(B, S)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mtp_matches_hf_oracle(dtype):
    """Bit-exact in both dtypes: same ops and cast points as HF's blocks, wired like the fork."""
    args, mtp, oracle = _mtp_and_oracle(dtype)
    h_main, e_next, pos = _inputs(args, dtype)
    rec = TensorRecorder()
    with hf_cpu_flash_attention(), torch.no_grad():
        h_in, expect = oracle(h_main, e_next, pos)
        out = mtp(h_main, e_next, pos, tap=rec)
    print(f"\n[{dtype}] MTP vs HF-block oracle: exact={torch.equal(out, expect)} max|d|={max_abs(out, expect):.3e}")
    # concat order [h_main, embed_norm(e)] and embed_norm placement, checked before the block
    assert torch.equal(rec.tensors["input_proj.out"], h_in)
    assert torch.equal(out, expect), max_abs(out, expect)
    assert out.dtype == dtype and out.shape == (B, S, args.hidden_size)


def _rewired(mtp, h_main, e_next, pos, variant):
    """``MotifMTP.forward`` with one wiring change, each a plausible porting mistake."""
    ne = mtp.embed_norm(e_next)
    parts = {
        "concat [embed_norm(e), h_main]": [ne, h_main],
        "no embed_norm": [h_main, e_next],
        "embed_norm on h_main": [mtp.embed_norm(h_main), e_next],
    }.get(variant, [h_main, ne])
    h = mtp.input_proj(torch.cat(parts, dim=-1))
    a = mtp.self_attn(mtp.input_layernorm(h), pos)
    h = a if variant == "no attention residual" else h + a
    f = mtp.mlp(mtp.post_attention_layernorm(h))
    h = f if variant == "no MLP residual" else h + f
    return h if variant == "no final_layernorm" else mtp.final_layernorm(h)


def test_mtp_oracle_check_is_sensitive_to_wiring():
    """Guard against a vacuous comparison: every wiring mistake moves the fp32 output by > 10% and every
    attention-config mistake by > 1e-4 relative (fp32 round-off is ~1e-7), so the exact comparison above catches
    each of them. The unmodified rewiring reproduces the oracle exactly."""
    args, mtp, oracle = _mtp_and_oracle(torch.float32)
    h_main, e_next, pos = _inputs(args, torch.float32)
    with hf_cpu_flash_attention(), torch.no_grad():
        _, expect = oracle(h_main, e_next, pos)
    scale = float(expect.abs().max())
    rel = lambda out: max_abs(out, expect) / scale  # noqa: E731
    with torch.no_grad():
        assert rel(_rewired(mtp, h_main, e_next, pos, "as shipped")) == 0.0
        for variant in (
            "concat [embed_norm(e), h_main]",
            "no embed_norm",
            "embed_norm on h_main",
            "no attention residual",
            "no MLP residual",
            "no final_layernorm",
        ):
            assert rel(_rewired(mtp, h_main, e_next, pos, variant)) > 0.1, variant
        attn = mtp.self_attn
        for attr, value in (
            ("window", None),  # global attention instead of SWA
            ("window", attn.window - 1),  # 128 keys instead of 129
            ("is_swa", False),  # global (YaRN) RoPE table instead of plain swa_rope_theta RoPE
            ("scale", args.softmax_scale(0)),  # global-layer softmax scale (YaRN mscale^2)
        ):
            old = getattr(attn, attr)
            setattr(attn, attr, value)
            try:
                assert rel(mtp(h_main, e_next, pos)) > 1e-4, (attr, value)
            finally:
                setattr(attn, attr, old)
        assert rel(mtp(h_main, e_next, pos)) == 0.0


def test_mtp_attention_config_matches_hf_block():
    """Window, softmax scale and RoPE table of the MTP attention equal what HF derives from the fork's MTP config."""
    args, mtp, oracle = _mtp_and_oracle(torch.float32)
    h_main, e_next, pos = _inputs(args, torch.float32)
    with hf_cpu_flash_attention():
        oracle(h_main, e_next, pos)  # materializes the HF rotary tables (built on the meta device)
    hf_attn = oracle.block.self_attn
    assert not oracle.block.moe_enabled and not oracle.block.mhc_enabled
    assert mtp.self_attn.window == hf_attn.sliding_window == args.effective_sliding_window == 129
    assert mtp.self_attn.scale == hf_attn.scaling == args.head_dim**-0.5
    assert hf_attn.swa_rotary_emb is not None and not mtp.self_attn.uses_yarn
    assert torch.equal(mtp.self_attn.inv_freq(), hf_attn.swa_rotary_emb.inv_freq)
    assert not torch.equal(mtp.self_attn.inv_freq(), oracle.rotary.inv_freq)  # the ignored global (YaRN) table


@pytest.mark.parametrize("attn_mode", ["expanded", "absorbed"])
def test_mtp_decode_matches_full(attn_mode):
    """Prefill 20 tokens, then decode one token at a time through the MTP block's own latent cache, past the window."""
    args, mtp, _ = _mtp_and_oracle(torch.float32)
    h_main, e_next, pos = _inputs(args, torch.float32)
    n_pre = 20
    cache = LatentKVCache(B, S, args.kv_lora_rank, args.qk_rope_head_dim, torch.float32, mtp.self_attn.window)
    with torch.no_grad():
        full = mtp(h_main, e_next, pos)
        steps = [mtp(h_main[:, :n_pre], e_next[:, :n_pre], pos[:, :n_pre], cache, attn_mode)]
        steps += [
            mtp(h_main[:, t : t + 1], e_next[:, t : t + 1], pos[:, t : t + 1], cache, attn_mode)
            for t in range(n_pre, S)
        ]
    dec = torch.cat(steps, dim=1)
    assert max_abs(dec, full) <= 1e-5 * float(full.abs().max()), max_abs(dec, full)
