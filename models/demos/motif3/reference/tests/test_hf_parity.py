# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""(a) Reference vs the official HF ``modeling_motif.py`` on tiny random configs.

HF runs with ``_attn_implementation="flash_attention_2"`` routed to an exact CPU flash-attn shim
(``hf_reference.py``; window mapping verified against transformers 5.12.1). Identical random weights are loaded
into both models; full-sequence logits and every layer's 4-stream input are compared.

The shipped HF file is not stale w.r.t. the fork's training-parity audit (motif_docs.md): h_post coefficient 1.0,
YaRN inv_freq for global layers, routed-only bias clamp, window + 1 and the fp32 ``poly * up`` multiply are all
present, so the reference is compared against HF directly. Known HF deviations are tested explicitly below.
"""

import pytest
import torch

from models.demos.motif3.reference import tiny_random_args
from models.demos.motif3.reference.generate import MotifGenerator
from models.demos.motif3.reference.golden import TensorRecorder

from .common import make_ref_and_hf, max_abs, pcc, rand_ids
from .hf_reference import hf_cpu_flash_attention


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "sliding_window,seq_len,batch",
    [(8, 37, 2), (128, 150, 1), (128, 300, 1)],
    ids=["win9_S37_B2", "win129_S150", "win129_S300"],
)
def test_full_model_logits_match_hf(dtype, sliding_window, seq_len, batch):
    args = tiny_random_args(sliding_window=sliding_window)
    ref, hf, _ = make_ref_and_hf(args, seed=1, dtype=dtype)
    ids = rand_ids(args, batch, seq_len, seed=3)
    rec = TensorRecorder(lambda n: n.endswith(".x_in") or n == "logits")
    with hf_cpu_flash_attention(), torch.no_grad():
        out = hf(ids, use_cache=False, output_hidden_states=True)
        logits = ref(ids, tap=rec)
    print(
        f"\n[{dtype}] logits max|ref-hf| = {max_abs(logits, out.logits):.3e}, exact={torch.equal(logits, out.logits)}"
    )
    # per-layer 4-stream inputs (HF hidden_states[i] is the input of layer i)
    for i in range(args.num_hidden_layers):
        a, b = rec.tensors[f"layers.{i}.x_in"], out.hidden_states[i]
        if dtype == torch.float32:
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5, msg=f"layer {i} input")
        else:
            assert pcc(a, b) > 0.99999, f"layer {i} input pcc {pcc(a, b)}"
    if dtype == torch.float32:
        torch.testing.assert_close(logits, out.logits, atol=1e-5, rtol=1e-5)
    else:
        assert pcc(logits, out.logits) > 0.99999
        assert max_abs(logits, out.logits) < 5e-2


@pytest.mark.parametrize("knob", ["q_path_fp32=False", "mhc_mix_fp32=True"])
def test_precision_knobs_are_precision_only(knob):
    """The fork-precision knobs change only rounding: identical to HF in fp32, very close in bf16."""
    name, value = knob.split("=")
    args = tiny_random_args(sliding_window=8, **{name: value == "True"})
    ids = rand_ids(args, 1, 40, seed=4)
    for dtype, check in (
        (torch.float32, lambda a, b: max_abs(a, b) < 1e-4),
        (torch.bfloat16, lambda a, b: pcc(a, b) > 0.999),
    ):
        ref, hf, _ = make_ref_and_hf(args, seed=2, dtype=dtype)
        with hf_cpu_flash_attention(), torch.no_grad():
            hf_logits = hf(ids, use_cache=False).logits
            ref_logits = ref(ids)
        assert check(ref_logits, hf_logits), f"{knob} {dtype}: max {max_abs(ref_logits, hf_logits)}"


def test_parity_is_sensitive_to_semantics():
    """Guard against a vacuous comparison: stale variants (window without +1, h_post = 2 sigmoid, no window)
    must move the logits by far more than the parity tolerance."""
    args = tiny_random_args(sliding_window=8)
    ref, _, _ = make_ref_and_hf(args, seed=1)
    ids = rand_ids(args, 1, 40, seed=5)
    with torch.no_grad():
        base = ref(ids)
        attn = [layer.self_attn for layer in ref.model.layers.values()]
        for a in attn:
            if a.window is not None:
                a.window -= 1  # 128 instead of 129 keys
        assert max_abs(ref(ids), base) > 1e-2
        for a in attn:
            if a.window is not None:
                a.window = None  # no window at all
        assert max_abs(ref(ids), base) > 1e-2
        for a, layer in zip(attn, ref.model.layers.values()):
            a.window = args.attention_window(layer.layer_idx)
            layer.mhc_attn.h_post_coeff = layer.mhc_ffn.h_post_coeff = 2.0  # stale "2 * sigmoid"
        assert max_abs(ref(ids), base) > 1e-2
        for layer in ref.model.layers.values():
            layer.mhc_attn.h_post_coeff = layer.mhc_ffn.h_post_coeff = 1.0
        assert torch.equal(ref(ids), base)


def test_hf_rope_tables_are_yarn_global_plain_swa():
    """HF (transformers 5.12.1) really uses YaRN on global layers and plain RoPE on SWA layers, like the reference
    and the fork (the fork's note that HF resets inv_freq to plain RoPE refers to an older HF revision)."""
    args = tiny_random_args(sliding_window=8, swa_rope_theta=5000.0)
    ref, hf, _ = make_ref_and_hf(args, seed=1)
    ids = rand_ids(args, 1, 12, seed=6)
    with hf_cpu_flash_attention(), torch.no_grad():
        torch.testing.assert_close(ref(ids), hf(ids, use_cache=False).logits, atol=1e-5, rtol=1e-5)
    assert torch.equal(hf.model.rotary_emb.inv_freq, ref.model.layers["0"].self_attn.inv_freq())
    assert torch.equal(hf.model.layers[1].self_attn.swa_rotary_emb.inv_freq, ref.model.layers["1"].self_attn.inv_freq())
    assert not torch.equal(ref.model.layers["0"].self_attn.inv_freq(), ref.model.layers["1"].self_attn.inv_freq())


def test_known_hf_bug_seq_len_equal_to_num_heads():
    """KNOWN HF DEVIATION (documented in README): ``MotifGDLAttention`` decides whether the attention output is
    (B, H, S, D) by ``attn_out.shape[1] == num_heads`` (modeling_motif.py:746). flash-attn returns (B, S, H, D), so a
    prefill with exactly S == num_attention_heads tokens (80 for Motif-3) is silently mis-transposed. The reference
    does not have this ambiguity; HF agrees with it for every other length."""
    args = tiny_random_args(sliding_window=8)
    ref, hf, _ = make_ref_and_hf(args, seed=1)
    H = args.num_attention_heads
    with hf_cpu_flash_attention(), torch.no_grad():
        for S in (H - 1, H + 1):
            ids = rand_ids(args, 1, S, seed=7)
            torch.testing.assert_close(ref(ids), hf(ids, use_cache=False).logits, atol=1e-5, rtol=1e-5)
        ids = rand_ids(args, 1, H, seed=7)
        assert max_abs(ref(ids), hf(ids, use_cache=False).logits) > 1e-2


def test_hf_decode_cache_pitfall_and_reference_decode():
    """HF decode is only valid with a config-less ``DynamicCache()``; ``DynamicCache(config=...)`` (what
    ``generate()`` builds) turns every layer into a (sliding_window - 1)-token sliding layer and diverges once the
    context exceeds the window (study §3.12). The reference decode matches the full forward everywhere."""
    from transformers.cache_utils import DynamicCache

    args = tiny_random_args(sliding_window=8)
    ref, hf, _ = make_ref_and_hf(args, seed=1)
    T, P = 40, 4
    ids = rand_ids(args, 1, T, seed=8)
    with hf_cpu_flash_attention(), torch.no_grad():
        full = hf(ids, use_cache=False).logits[0]

        def hf_decode(cache):
            outs = [hf(ids[:, :P], past_key_values=cache, use_cache=True).logits[0]]
            outs += [hf(ids[:, t : t + 1], past_key_values=cache, use_cache=True).logits[0] for t in range(P, T)]
            return torch.cat(outs, 0)

        good = hf_decode(DynamicCache())
        bad = hf_decode(DynamicCache(config=hf.config))
    gen = MotifGenerator(ref, 1, T, attn_mode="absorbed")
    ref_dec = torch.cat([gen.prefill(ids[:, :P])[0]] + [gen.decode(ids[:, t])[None, 0] for t in range(P, T)], 0)
    torch.testing.assert_close(good, full, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ref_dec, full, atol=1e-4, rtol=1e-4)
    err = (bad - full).abs().amax(-1)
    window = args.effective_sliding_window
    assert err[: window - 1].max() < 1e-4  # identical while the context fits the (wrong) window
    assert err[window:].max() > 1e-2  # wrong past it
