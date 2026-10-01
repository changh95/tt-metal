# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""(c) KV-cache decode == full-sequence forward past the sliding window (300 tokens, window 129, tiny config)."""

import pytest
import torch

from models.demos.motif3.reference import (
    GDLAttention,
    attention_mask,
    build_random_model,
    random_state_dict,
    tiny_random_args,
)
from models.demos.motif3.reference.generate import MotifGenerator
from models.demos.motif3.reference.weights import round_state_dict

from .common import max_abs, pcc, rand_ids

T = 300


@pytest.fixture(scope="module")
def model_and_full():
    args = tiny_random_args()
    assert args.effective_sliding_window == 129
    model = build_random_model(args, seed=5)
    ids = rand_ids(args, 1, T, seed=9)
    with torch.no_grad():
        full = model(ids, attn_mode="expanded")
    return model, ids, full


@pytest.mark.parametrize("attn_mode", ["expanded", "absorbed"])
def test_decode_matches_full_forward_past_window(model_and_full, attn_mode):
    model, ids, full = model_and_full
    n_pre = 50
    gen = MotifGenerator(model, batch_size=1, max_seq_len=T, attn_mode=attn_mode)
    logits = [gen.prefill(ids[:, :n_pre])[0]]
    logits += [gen.decode(ids[:, t])[0][None] for t in range(n_pre, T)]
    dec = torch.cat(logits, 0)
    err = (dec - full[0]).abs().amax(-1)
    print(f"\n[{attn_mode}] decode vs full: max {err.max():.2e}, positions >= 129: {err[129:].max():.2e}")
    assert err.max() < 1e-4
    assert int(gen.positions[0]) == T


def test_chunked_prefill_matches_full(model_and_full):
    model, ids, full = model_and_full
    gen = MotifGenerator(model, 1, T, attn_mode="expanded")
    chunks = [gen.prefill(ids[:, a:b]) for a, b in ((0, 100), (100, 230), (230, T))]
    assert max_abs(torch.cat(chunks, 1), full) < 1e-4


def test_batched_decode_with_heterogeneous_positions(model_and_full):
    """Users of one batch at different positions (one past the window, one inside it) decode together exactly like
    separate single-user generators."""
    model, ids, _ = model_and_full
    lens = (140, 60)
    steps = 5
    gen = MotifGenerator(model, batch_size=2, max_seq_len=T, attn_mode="absorbed")
    for b, n in enumerate(lens):
        gen.prefill(ids[0, :n], user=b)
    singles = []
    for n in lens:
        g1 = MotifGenerator(model, 1, T, attn_mode="absorbed")
        g1.prefill(ids[:, :n])
        singles.append(g1)
    for s in range(steps):
        tok = torch.stack([ids[0, lens[0] + s], ids[0, lens[1] + s]])
        batched = gen.decode(tok)
        for b in range(2):
            single = singles[b].decode(tok[b : b + 1])
            assert max_abs(batched[b], single[0]) < 1e-5
    assert gen.positions.tolist() == [lens[0] + steps, lens[1] + steps]


def test_generate_greedy_is_teacher_forced_argmax(model_and_full):
    model, ids, _ = model_and_full
    prompt = ids[:, :120]
    gen = MotifGenerator(model, 1, T, attn_mode="absorbed")
    new = gen.generate(prompt, max_new_tokens=20, eos_token_ids=())[0]
    seq = torch.cat([prompt, torch.tensor([new])], dim=1)
    with torch.no_grad():
        full = model(seq)
    # logits at position prompt_len - 1 + k predict new[k]
    preds = full[0, prompt.shape[1] - 1 : -1].argmax(-1).tolist()
    assert len(new) == 20 and preds == new


def test_mask_window_semantics():
    """129 keys including the current one on SWA layers ([p-128, p]); p+1 keys on global layers."""
    q = torch.arange(300)[None]
    k = torch.arange(300)
    swa = attention_mask(q, k, 129)[0, 0]
    glob = attention_mask(q, k, None)[0, 0]
    p = torch.arange(300)
    assert torch.equal(swa.sum(-1), torch.clamp(p + 1, max=129))
    assert torch.equal(glob.sum(-1), p + 1)
    assert bool(swa[200, 72]) and not bool(swa[200, 71]) and not bool(swa[200, 201])
    # decode: one query at position 299 against 300 cached keys
    dec = attention_mask(torch.tensor([[299]]), k, 129)[0, 0, 0]
    assert dec.sum() == 129 and bool(dec[171]) and not bool(dec[170])


def test_generate_sampling_is_seeded_and_valid(model_and_full):
    model, ids, _ = model_and_full
    outs = []
    for _ in range(2):
        gen = MotifGenerator(model, 1, T, attn_mode="absorbed")
        g = torch.Generator().manual_seed(123)
        outs.append(
            gen.generate(ids[:, :20], 8, eos_token_ids=(), sample=True, temperature=1.0, top_p=0.95, generator=g)[0]
        )
    assert outs[0] == outs[1] and len(outs[0]) == 8
    assert all(0 <= t < model.args.vocab_size for t in outs[0])


@pytest.mark.parametrize("attn_mode", ["expanded", "absorbed"])
def test_cache_dtype_is_storage_precision_only(attn_mode, monkeypatch):
    """``cache_dtype`` != model dtype works in both attention forms and only changes how the cache stores entries.
    An fp32 cache on a bf16 model is lossless (bit-identical to the bf16 cache); a bf16 cache on an fp32 model equals
    the fp32 model with its cache entries (kv_norm(c), roped k_pe) rounded to bf16."""
    args = tiny_random_args(sliding_window=8)
    sd = round_state_dict(random_state_dict(args, seed=1), torch.bfloat16)
    T, P = 16, 6  # window 9
    ids = rand_ids(args, 2, T, seed=2)

    def run(model, cache_dtype=None):
        gen = MotifGenerator(model, 2, T, attn_mode=attn_mode, cache_dtype=cache_dtype)
        assert gen.cache[0].c.dtype == (cache_dtype or model.dtype)
        steps = [gen.prefill(ids[:, :P])] + [gen.decode(ids[:, t])[:, None] for t in range(P, T)]
        return torch.cat(steps, dim=1)

    bf16_model = build_random_model(args, dtype=torch.bfloat16, state_dict=sd)
    assert torch.equal(run(bf16_model, torch.float32), run(bf16_model))

    fp32_model = build_random_model(args, dtype=torch.float32, state_dict=sd)
    bf16_cache, fp32_cache = run(fp32_model, torch.bfloat16), run(fp32_model)
    assert not torch.equal(bf16_cache, fp32_cache) and pcc(bf16_cache, fp32_cache) > 0.999  # rounds, slightly
    project_kv = GDLAttention.project_kv

    def project_kv_rounded(self, x, cos, sin):
        c, k_pe = project_kv(self, x, cos, sin)
        return c.bfloat16().float(), k_pe.bfloat16().float()

    monkeypatch.setattr(GDLAttention, "project_kv", project_kv_rounded)
    assert torch.equal(bf16_cache, run(fp32_model))


def test_generator_rejects_unknown_attn_mode():
    model = build_random_model(tiny_random_args(num_hidden_layers=1), seed=0)
    with pytest.raises(ValueError, match="attention mode"):
        MotifGenerator(model, 1, 8, attn_mode="mla")
