# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the MTP drafter pieces (no device, no ttnn ops).

* test_chain_drafts_schedule / test_chain_with_verify_controller: the chained draft schedule (verify_grid.chain_drafts)
  drives a CPU stand-in of the MTP head -- a per-user position-indexed KV of (token) writes -- and the VerifyController
  with the CpuGreedyOracle: the head's KV writes land at P_s - 1 + j, the drafts are rows 1..k, a perfect head yields
  k accepted drafts per step, a noisy head still reproduces the plain greedy stream (draft-independence).
* test_mtp_reference_vs_hf_blocks (needs the checkpoint on disk, ~8 GB host RAM): MTPHostReference (hand-written
  fp32 torch, the device module's oracle) vs the SAME head assembled from HF transformers' own Qwen3.5 blocks
  (Qwen3_5DecoderLayer(full_attention) + Qwen3_5RMSNorm + Qwen3_5TextRotaryEmbedding, loaded with the mtp.* weights, the
  structure of vLLM's Qwen3_5MultiTokenPredictor.forward) on a short prompt with random main hidden states: full
  sequence and incremental (prefill + chained steps) must agree.

  pytest models/demos/blackhole/qwen36/tests/test_mtp_cpu.py -s
"""
import glob
import os
import random

import pytest
import torch

from models.demos.blackhole.qwen36.tt import verify_grid as vg

SNAP = os.environ.get(
    "MTP_CKPT", next(iter(glob.glob("/home/eslim/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/")), "")
)


class CpuMtpStub:
    """A CPU 'MTP head': its draft for the token after position p+1 is next_fn(prefix up to p+1) when perfect, else
    random with probability noise. Records every KV write position per user (the schedule under test)."""

    def __init__(self, next_fn, prompts, k, noise=0.0, seed=0):
        self.next_fn = next_fn
        self.prompts = [list(p) for p in prompts]
        self.noise = noise
        self.rng = random.Random(seed)
        self.kv_writes = [[] for _ in prompts]
        self.k = k
        self.ctrl = None  # set by the test: reads the committed streams for the "true prefix"

    def run_step(self, tokens, positions):
        out = []
        for s, (t, p) in enumerate(zip(tokens, positions)):
            self.kv_writes[s].append(int(p))
            # the true prefix up to position p (exclusive of the token at p+1 = t): committed stream + chain tokens
            prefix = self._prefix(s, int(p)) + [int(t)]
            d = int(self.next_fn(prefix))
            if self.noise and self.rng.random() < self.noise:
                d = (d + 1 + self.rng.randrange(1000)) % 250000
            out.append(d)
        return out

    def _prefix(self, s, p):
        # tokens at positions 0..p: the prompt + committed stream (positions >= len(prompt)) + the current chain
        full = self.prompts[s] + self.ctrl.committed[s][:-1] + [self.ctrl.last[s]] + self._chain.get(s, [])
        return full[: p + 1]

    def draft(self, ctrl):
        self.ctrl = ctrl
        self._chain = {}
        w = ctrl.users

        def obs(phase, j, toks, poss, drafts):
            if phase == "post":
                for s in range(w):
                    self._chain.setdefault(s, []).append(drafts[s])

        return vg.chain_drafts(self.run_step, w, self.k, ctrl.last, ctrl.positions, observer=obs)


def _next_fn(prefix):
    # a deterministic pseudo model: next token = hash of the last 3 tokens
    h = 0
    for t in prefix[-3:]:
        h = (h * 1000003 + int(t) + 7) % 100003
    return h % 5000


@pytest.mark.parametrize("w,k", [(1, 1), (1, 3), (4, 2), (8, 3)])
def test_chain_drafts_schedule(w, k):
    """Step j writes KV at P_s - 1 + j with token d_j (d_0 = t'_s); the drafts are the returned lists in step order."""
    last = [100 + s for s in range(w)]
    positions = [10 + 3 * s for s in range(w)]
    seen = []

    def run_step(tokens, poss):
        seen.append((list(tokens), list(poss)))
        return [t + 1 for t in tokens]

    drafts = vg.chain_drafts(run_step, w, k, last, positions)
    assert len(seen) == k
    for j, (toks, poss) in enumerate(seen):
        assert poss == [p - 1 + j for p in positions]
        assert toks == [t + j for t in last]
    assert drafts == [[last[s] + 1 + j for j in range(k)] for s in range(w)]


@pytest.mark.parametrize("w,k,noise", [(1, 1, 0.0), (1, 3, 0.0), (4, 2, 0.0), (8, 3, 0.4), (3, 3, 1.0)])
def test_chain_with_verify_controller(w, k, noise):
    """Drafter + VerifyController(CpuGreedyOracle) reproduce the plain greedy stream for any head quality; a perfect
    head accepts every draft; the head's KV writes are exactly the positions P_s - 1 + j of every step."""
    T = k + 1
    prompts = [[1 + s, 2, 3 + s, 4] for s in range(w)]
    n_new = 24
    # plain greedy streams
    greedy = []
    for s in range(w):
        seq = list(prompts[s])
        for _ in range(n_new + T):
            seq.append(_next_fn(seq))
        greedy.append(seq[len(prompts[s]) :])
    oracle = vg.CpuGreedyOracle(_next_fn, prompts, T)
    first = [_next_fn(p) for p in prompts]
    ctrl = vg.VerifyController(T=T, run=oracle, positions=[len(p) for p in prompts], last=first)
    head = CpuMtpStub(_next_fn, prompts, k, noise=noise, seed=w * 7 + k)
    expected_writes = [[] for _ in range(w)]
    drafts = head.draft(ctrl)
    for s in range(w):
        expected_writes[s] += [ctrl.positions[s] - 1 + j for j in range(k)]
    steps = 0
    while min(len(c) for c in ctrl.committed) < n_new:
        accepts = ctrl.step(drafts)
        if noise == 0.0:
            assert accepts == [k] * w, accepts
        drafts = head.draft(ctrl)
        for s in range(w):
            expected_writes[s] += [ctrl.positions[s] - 1 + j for j in range(k)]
        steps += 1
    for s in range(w):
        n = min(len(ctrl.committed[s]), len(greedy[s]))
        assert ctrl.committed[s][:n] == greedy[s][:n], f"user {s} diverged from greedy"
        assert head.kv_writes[s] == expected_writes[s]
    if noise == 0.0:  # a perfect head commits k+1 tokens per step
        assert all(len(c) == 1 + steps * T for c in ctrl.committed)


@pytest.mark.skipif(not SNAP or not os.path.isdir(SNAP), reason="checkpoint not on disk")
def test_mtp_reference_vs_hf_blocks():
    """MTPHostReference vs the head assembled from HF transformers' Qwen3.5 blocks (vLLM Qwen3_5MultiTokenPredictor
    structure): fc(cat(norm_e(embed(x_{i+1})), norm_h(h_i))) -> one full-attention decoder layer -> norm -> lm_head."""
    from transformers.models.qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5RMSNorm,
        Qwen3_5TextRotaryEmbedding,
    )

    from models.demos.blackhole.qwen36.tt.mtp_head import MTPHostReference
    from models.demos.blackhole.qwen36.tt.weight_mapping import load_qwen36_mtp_state_dict

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig.from_pretrained(SNAP)
    cfg._attn_implementation = "eager"
    cfg.layer_types = ["full_attention"]  # the MTP layer

    class Args:
        dim = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        rope_head_dim = int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"])
        rope_theta = cfg.rope_parameters["rope_theta"]
        norm_eps = cfg.rms_norm_eps
        vocab_size = cfg.vocab_size

    mine = MTPHostReference(SNAP, Args)
    sd = load_qwen36_mtp_state_dict(SNAP, 0)
    with torch.no_grad():
        layer = Qwen3_5DecoderLayer(cfg, 0).float()
        layer.load_state_dict({k[len("layers.0.") :]: v.float() for k, v in sd.items() if k.startswith("layers.0.")})
        norms = {}
        for key in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
            n = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            n.weight.copy_(sd[f"mtp.{key}.weight"].float())
            norms[key] = n
        fc = torch.nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        fc.weight.copy_(sd["mtp.fc.weight"].float())
        rotary = Qwen3_5TextRotaryEmbedding(cfg)

        def hf_forward(tokens, H, positions):
            S = len(tokens)
            e = mine.embed[torch.as_tensor(tokens)].float()[None]  # [1,S,dim]
            x = fc(torch.cat([norms["pre_fc_norm_embedding"](e), norms["pre_fc_norm_hidden"](H.float()[None])], -1))
            pos = torch.as_tensor(positions)[None]
            cos, sin = rotary(x, pos)
            mask = torch.full((S, S), float("-inf")).triu(1)[None, None]
            y = layer(x, position_embeddings=(cos, sin), attention_mask=mask, position_ids=pos)
            out = norms["norm"](y)[0]
            return out @ mine.lm_head.float().T, out

        S = 12
        tokens = torch.randint(0, cfg.vocab_size, (S + 3,)).tolist()
        H = (torch.randn(S + 3, cfg.hidden_size) * 1.5).to(torch.bfloat16)
        hf_logits, hf_out = hf_forward(tokens, H, list(range(S + 3)))
        # mine: full sequence in one shot
        st = mine.new_state()
        my_logits, my_out = mine.forward(st, tokens, H, list(range(S + 3)))
        # mine: prefill S rows, then 3 chained single-token steps (the drafter's use)
        st2 = mine.new_state()
        lg_p, _ = mine.forward(st2, tokens[:S], H[:S], list(range(S)))
        inc = [lg_p]
        for j in range(3):
            lg_j, _ = mine.forward(st2, [tokens[S + j]], H[S + j : S + j + 1], [S + j])
            inc.append(lg_j)
        my_inc = torch.cat(inc)

    def pcc(a, b):
        a = a.reshape(-1) - a.mean()
        b = b.reshape(-1) - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()))

    d_full = float((hf_logits - my_logits).abs().max())
    d_inc = float((hf_logits - my_inc).abs().max())
    scale = float(hf_logits.abs().max())
    print(
        f"MTP_CPU_REF hf-vs-mine max|d| full {d_full:.3e} incremental {d_inc:.3e} (|logits| max {scale:.2f}), "
        f"pcc full {pcc(hf_logits, my_logits):.7f} inc {pcc(hf_logits, my_inc):.7f}, "
        f"argmax agree full {(hf_logits.argmax(-1) == my_logits.argmax(-1)).float().mean():.3f} "
        f"inc {(hf_logits.argmax(-1) == my_inc.argmax(-1)).float().mean():.3f}"
    )
    assert d_full < 1e-2 * max(1.0, scale) and d_inc < 1e-2 * max(1.0, scale)
    assert torch.equal(hf_logits.argmax(-1), my_logits.argmax(-1)) and torch.equal(
        hf_logits.argmax(-1), my_inc.argmax(-1)
    )
    assert float((hf_out - my_out).abs().max()) < 1e-2 * float(hf_out.abs().max())
