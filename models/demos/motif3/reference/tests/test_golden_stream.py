# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""golden_stream.py: the streaming, resumable full-model golden equals the reference's ordinary full forward.

* Tiny random 6-layer config (every layer kind), saved as a bf16 safetensors checkpoint with one shard per layer: a
  streaming run interrupted after layer 2 and resumed for layers 3-5 reproduces ``MotifForCausalLM.forward`` of every
  prompt bit for bit (embedding, each layer's 4-stream state, stream mean, final norm, logits -> top-k / argmax /
  target log-probs), in bf16 and in fp32, and writes the same tensors as an uninterrupted run.
* Refusals: a fresh run into a used directory, a resume with a gap / another dtype / other prompts; layers whose
  shards are incomplete or not listed in ``.download_state.json`` are not run.
* Real weights (skipped unless layers 0-3, embed, norm and lm_head are local): layers 0-1, then a resumed 2-3 with
  the early-exit head, on two prompts of the real set, equal ``load_reference_model(layer_ids=range(4))``.
* The prompt set: ``rendered.json`` is what ``render_prompt_set`` makes of ``messages.json``; ``chat_default`` is
  ``golden.DEFAULT_PROMPT_MESSAGES``; per-token roles follow the chat template.
"""

import json
import shutil

import pytest
import torch

from models.demos.motif3.reference import load_reference_model, random_state_dict, reference_to_hf_state_dict
from models.demos.motif3.reference.config import tiny_random_args
from models.demos.motif3.reference.golden import DEFAULT_PROMPT_MESSAGES, TensorRecorder
from models.demos.motif3.reference.golden_stream import (
    DEFAULT_MESSAGES,
    DEFAULT_PROMPT_SET,
    ROLE_NAMES,
    StreamPrompt,
    head_paths,
    load_prompt_set,
    load_states,
    load_tensors,
    render_prompt_set,
    resolve_layers,
    run_stream,
    state_path,
)
from models.demos.motif3.reference.tokenizer import encode_chat, load_tokenizer
from models.demos.motif3.reference.weights import DEFAULT_WEIGHTS_DIR, MissingWeightsError

from .common import max_abs, real_checkpoint

pytestmark = pytest.mark.timeout(1800)

TAPS = ("embed", "final.stream_mean", "final.norm", "logits")


def _quiet(msg):
    pass


def _identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise-equal values, dtype and shape (NaN == NaN: target_logit / target_logprob are NaN at the last position)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        return torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(0.0), b.nan_to_num(0.0))
    return torch.equal(a, b)


def _write_tiny_checkpoint(path, args, sd_ref):
    """HF-named bf16 safetensors checkpoint laid out like the real one: one shard per decoder layer plus one for
    embed / final norm / lm_head, an index and a config.json."""
    from safetensors.torch import save_file

    hf_sd = reference_to_hf_state_dict(sd_ref)
    groups = {}
    for name, t in hf_sd.items():
        key = int(name.split(".")[2]) if name.startswith("model.layers.") else "head"
        groups.setdefault(key, {})[name] = t.to(torch.bfloat16).contiguous()
    weight_map = {}
    for s, key in enumerate(sorted(groups, key=str)):
        fn = f"model-{s + 1:05d}-of-{len(groups):05d}.safetensors"
        save_file(groups[key], str(path / fn))
        weight_map.update({n: fn for n in groups[key]})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    cfg = args.to_hf_config_dict()
    cfg["eos_token_id"] = list(args.eos_token_ids)
    (path / "config.json").write_text(json.dumps(cfg))
    return weight_map


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_ckpt")
    args = tiny_random_args()
    weight_map = _write_tiny_checkpoint(path, args, random_state_dict(args, seed=11))
    return path, args, weight_map


def _tiny_prompts(args):
    """Three prompts: longer than the 129-key window, short, and just past it; one has an 'assistant' region."""
    g = torch.Generator().manual_seed(5)
    out = []
    for name, n in (("long", 150), ("short", 37), ("window", 131)):
        ids = torch.randint(0, args.vocab_size, (n,), generator=g).tolist()
        roles = [0] * n if name != "long" else [2] * 90 + [3] * 60
        out.append(StreamPrompt(name, ids, roles))
    return out


@torch.no_grad()
def _reference_taps(model, prompt):
    rec = TensorRecorder(lambda n: n.endswith(".x_out") or n in TAPS)
    model(torch.tensor([prompt.ids]), tap=rec)
    return rec.tensors


def _assert_head_matches(lt, ht, prompt, ref, topk=32):
    """Stored head outputs of ``prompt`` vs the reference forward's logits / final norm / stream mean."""
    n = prompt.name
    logits = ref["logits"][0]
    assert torch.equal(ht[f"{n}.final_hidden"], ref["final.norm"].float())
    assert torch.equal(ht[f"{n}.stream_mean"], ref["final.stream_mean"].float())
    top_v, top_i = torch.topk(logits, topk, dim=-1)
    assert torch.equal(lt[f"{n}.topk_ids"], top_i) and torch.equal(lt[f"{n}.topk_logits"], top_v)
    assert lt[f"{n}.topk_logits"].dtype == torch.float32
    assert torch.equal(lt[f"{n}.argmax"], logits.argmax(-1))
    assert torch.equal(lt[f"{n}.argmax_ties"], (logits == logits.max(-1, keepdim=True).values).sum(-1))
    target = torch.tensor(prompt.ids[1:])
    S = len(prompt.ids)
    assert torch.equal(lt[f"{n}.target_ids"], torch.tensor(prompt.ids[1:] + [-1]))
    t_logit = logits[torch.arange(S - 1), target]
    lse = torch.logsumexp(logits, -1)
    assert torch.equal(lt[f"{n}.target_logit"][:-1], t_logit) and torch.equal(lt[f"{n}.logsumexp"], lse)
    assert torch.equal(lt[f"{n}.target_logprob"][:-1], t_logit - lse[:-1])
    # independent formulation; fp32 sums over the vocab round differently (~3e-5 at V = 220160)
    logp = torch.log_softmax(logits, -1)[torch.arange(S - 1), target]
    torch.testing.assert_close(lt[f"{n}.target_logprob"][:-1], logp, atol=2e-4, rtol=0)
    assert torch.isnan(lt[f"{n}.target_logprob"][-1]) and lt[f"{n}.target_rank"][-1] == -1
    assert torch.equal(lt[f"{n}.target_rank"][:-1], (logits[:-1] > t_logit[:, None]).sum(-1))
    in_asst = torch.tensor([r in (3, 4) for r in (prompt.roles or [0] * S)][1:] + [False])
    assert torch.equal(lt[f"{n}.target_in_assistant"], in_asst)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_tiny_stream_with_resume_equals_full_forward(tiny, tmp_path, dtype):
    path, args, _ = tiny
    prompts = _tiny_prompts(args)
    out = tmp_path / "resumed"
    m1 = run_stream(out, "0-2", prompts=prompts, ckpt_dir=path, dtype=dtype, save_layers=None, log=_quiet)
    assert m1["last_layer"] == 2 and m1["layers_done"] == [0, 1, 2] and not m1["heads"]
    m2 = run_stream(out, "auto", resume=True, ckpt_dir=path, save_layers=None, log=_quiet)
    assert m2["last_layer"] == 5 and list(m2["heads"]) == ["5"] and not m2["heads"]["5"]["early_exit"]
    assert m2["mode"]["dtype"] == {torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype]
    assert m2["saved_layers"] == list(range(6)) and len(m2["runs"]) == 2

    model = load_reference_model(path, layer_ids=None, dtype=dtype, lazy_experts=True)
    embed = load_tensors(out / "states" / "embed.safetensors")[0]
    lt, lmeta = load_tensors(head_paths(out, 5)[0])
    ht, _ = load_tensors(head_paths(out, 5)[1])
    assert lmeta["after_layer"] == 5 and lmeta["topk"] == 32
    states = {L: load_states(out, L) for L in range(6)}
    worst = 0.0
    for p in prompts:
        ref = _reference_taps(model, p)
        assert torch.equal(embed[p.name], ref["embed"])
        for L in range(6):
            got, exp = states[L][p.name], ref[f"layers.{L}.x_out"]
            assert got.dtype == dtype and got.shape == (1, len(p.ids), 4, args.hidden_size)
            worst = max(worst, max_abs(got, exp))
            assert torch.equal(got, exp), (p.name, L, max_abs(got, exp))
        _assert_head_matches(lt, ht, p, ref)
    print(f"\n[{dtype}] tiny 6-layer stream (resumed after layer 2) vs full forward: max|diff| = {worst:.1e}")

    # an uninterrupted run writes the same tensors
    straight = tmp_path / "straight"
    run_stream(straight, "all", prompts=prompts, ckpt_dir=path, dtype=dtype, save_layers=None, log=_quiet)
    for rel in [f"states/after_layer_{L:02d}.safetensors" for L in range(6)] + [
        "resume/state.safetensors",
        "final/logits_after_layer_05.safetensors",
        "final/hidden_after_layer_05.safetensors",
    ]:
        a, b = load_tensors(out / rel)[0], load_tensors(straight / rel)[0]
        assert a.keys() == b.keys() and all(_identical(a[k], b[k]) for k in a), rel


def test_tiny_stream_refusals_and_overwrite(tiny, tmp_path):
    path, args, _ = tiny
    prompts = _tiny_prompts(args)
    out = tmp_path / "out"
    run_stream(out, "0-1", prompts=prompts, ckpt_dir=path, log=_quiet)
    with pytest.raises(FileExistsError):
        run_stream(out, "0-1", prompts=prompts, ckpt_dir=path, log=_quiet)
    with pytest.raises(ValueError, match="next layer to process is 2"):
        run_stream(out, "3-5", resume=True, ckpt_dir=path, log=_quiet)
    with pytest.raises(ValueError, match="dtype"):
        run_stream(out, "2", resume=True, ckpt_dir=path, dtype=torch.float32, log=_quiet)
    with pytest.raises(ValueError, match="prompts"):
        run_stream(out, "2", resume=True, ckpt_dir=path, prompts=prompts[:1], log=_quiet)
    with pytest.raises(ValueError, match="contiguous"):
        run_stream(out, "2,4", resume=True, ckpt_dir=path, log=_quiet)
    # nothing to do, no head requested: a no-op that keeps the checkpoint
    m = run_stream(out, [], resume=True, ckpt_dir=path, log=_quiet)
    assert m["last_layer"] == 1 and not m["heads"]
    # forced early-exit head on the existing checkpoint (a logit lens after layer 1)
    m = run_stream(out, [], resume=True, ckpt_dir=path, final_head=True, log=_quiet)
    assert m["heads"]["1"]["early_exit"] and head_paths(out, 1)[0].exists()
    # a fresh run with overwrite removes the old run's files
    m = run_stream(out, "0", prompts=prompts[:1], ckpt_dir=path, overwrite=True, log=_quiet)
    assert m["layers_done"] == [0] and not state_path(out, 1).exists() and not head_paths(out, 1)[0].exists()
    assert json.loads((out / "prompts.json").read_text())["prompts"][0]["name"] == prompts[0].name


def test_tiny_stream_runs_only_downloaded_layers(tiny, tmp_path):
    src, args, weight_map = tiny
    path = tmp_path / "ckpt"
    shutil.copytree(src, path)
    (path / ".download_state.json").write_text(json.dumps({"complete_layers": [0, 1, 2]}))
    prompts = _tiny_prompts(args)[1:]
    out = tmp_path / "out"
    with pytest.raises(MissingWeightsError, match=r"\[3, 4, 5\]"):
        run_stream(out, "0-5", prompts=prompts, ckpt_dir=path, log=_quiet)
    m = run_stream(out, "auto", prompts=prompts, ckpt_dir=path, log=_quiet)
    assert m["last_layer"] == 2 and not m["heads"]
    # layers 3-5 "arrive", but layer 4's shard is still being written (truncated): auto stops after layer 3
    (path / ".download_state.json").write_text(json.dumps({"complete_layers": list(range(6))}))
    shard = path / weight_map["model.layers.4.self_attn.wq_a.weight"]
    data = shard.read_bytes()
    shard.write_bytes(data[: len(data) // 2])
    m = run_stream(out, "auto", resume=True, ckpt_dir=path, log=_quiet)
    assert m["last_layer"] == 3 and not m["heads"]
    shard.write_bytes(data)
    m = run_stream(out, "auto", resume=True, ckpt_dir=path, log=_quiet)
    assert m["last_layer"] == 5 and "5" in m["heads"]
    # the default save list keeps only the listed layers; the resume checkpoint holds the state after layer 5
    assert m["saved_layers"] == [0, 1, 2, 3, 4]
    last, meta = load_tensors(out / "resume" / "state.safetensors")
    ht = load_tensors(head_paths(out, 5)[1])[0]
    assert meta["last_layer"] == 5 and set(last) == {p.name for p in prompts}
    for p in prompts:
        assert torch.equal(ht[f"{p.name}.stream_mean"], last[p.name].mean(dim=2).float())


def test_resolve_layers():
    avail = lambda i: i < 36  # noqa: E731
    assert resolve_layers("0-35", 0, 53, avail) == list(range(36))
    assert resolve_layers("auto", 0, 53, avail) == list(range(36))
    assert resolve_layers("auto", 36, 53, avail) == []
    assert resolve_layers("36-52", 36, 53) == list(range(36, 53))
    assert resolve_layers("36-", 36, 53) == list(range(36, 53))
    assert resolve_layers("all", 50, 53) == [50, 51, 52]
    assert resolve_layers([7, 8], 7, 53) == [7, 8]
    with pytest.raises(ValueError, match="next layer to process is 36"):
        resolve_layers("37-52", 36, 53)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_layers("36-53", 36, 53)
    with pytest.raises(MissingWeightsError):
        resolve_layers("30-40", 30, 53, avail)


# ---- real weights ----------------------------------------------------------------------------------------------
REAL_STREAM_LAYERS = (0, 1, 2, 3)  # global/dense, swa/dense, swa/moe, swa/moe
REAL_PROMPTS = ("chat_default", "multi_turn_chat")  # 145 and 234 tokens: both past the 129-key window


def test_real_layers_0_3_stream_equals_prefix_forward(tmp_path):
    ckpt = real_checkpoint(REAL_STREAM_LAYERS)
    prompts = load_prompt_set(DEFAULT_PROMPT_SET, REAL_PROMPTS)
    out = tmp_path / "out"
    run_stream(
        out, "0-1", prompts=prompts, prompt_set_path=DEFAULT_PROMPT_SET, ckpt_dir=ckpt.dir, save_layers=None, log=_quiet
    )
    m = run_stream(out, "2-3", resume=True, ckpt_dir=ckpt.dir, save_layers=None, final_head=True, log=_quiet)
    assert m["mode"]["dtype"] == "bf16" and m["mode"]["attn_mode"] == "expanded" and m["mode"]["q_path_fp32"]
    assert m["heads"]["3"]["early_exit"]
    frozen = json.loads((out / "prompts.json").read_text())
    assert [p["name"] for p in frozen["prompts"]] == list(REAL_PROMPTS) and "text" in frozen["prompts"][0]

    model = load_reference_model(
        ckpt.dir, layer_ids=REAL_STREAM_LAYERS, dtype=torch.bfloat16, lazy_experts=True, checkpoint=ckpt
    )
    lt, ht = load_tensors(head_paths(out, 3)[0])[0], load_tensors(head_paths(out, 3)[1])[0]
    worst = 0.0
    for p in prompts:
        ref = _reference_taps(model, p)
        for L in REAL_STREAM_LAYERS:
            got, exp = load_states(out, L)[p.name], ref[f"layers.{L}.x_out"]
            worst = max(worst, max_abs(got, exp))
            assert got.dtype == torch.bfloat16 and torch.equal(got, exp), (p.name, L, max_abs(got, exp))
        _assert_head_matches(lt, ht, p, ref)
    print(f"\nreal layers 0-3 (resumed after layer 1) vs prefix-model forward: max|diff| = {worst:.1e}")


# ---- prompt set ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tokenizer():
    if not (DEFAULT_WEIGHTS_DIR / "tokenizer.json").exists():
        pytest.skip("tokenizer files not available")
    return load_tokenizer()


def test_prompt_set_is_rendered_from_messages(tokenizer):
    on_disk = json.loads(DEFAULT_PROMPT_SET.read_text(encoding="utf-8"))
    fresh = render_prompt_set(DEFAULT_MESSAGES, out_dir=None, tokenizer=tokenizer)
    assert fresh["sha256"] == on_disk["sha256"], "prompts/rendered.json is stale: run --render-prompts"
    for a, b in zip(fresh["prompts"], on_disk["prompts"]):
        assert (a["name"], a["ids"], a["roles"], a["text"]) == (b["name"], b["ids"], b["roles"], b["text"])
        txt = (DEFAULT_PROMPT_SET.parent / f"{a['name']}.txt").read_text(encoding="utf-8")
        assert txt == a["text"] + "\n"
    assert on_disk["role_names"] == list(ROLE_NAMES)


def test_prompt_set_contents(tokenizer):
    doc = json.loads(DEFAULT_PROMPT_SET.read_text(encoding="utf-8"))
    by_name = {p["name"]: p for p in doc["prompts"]}
    assert list(by_name) == [
        "chat_default",
        "en_technical",
        "ko_passage",
        "math_word_problem",
        "python_code",
        "multi_turn_chat",
    ]
    default = by_name["chat_default"]
    assert default["messages"] == DEFAULT_PROMPT_MESSAGES
    assert default["ids"] == encode_chat(DEFAULT_PROMPT_MESSAGES, tokenizer)  # golden.export_real_goldens' prompt
    assert default["ids"][-3:] == [5, 4, 11] and default["roles"][-1] == ROLE_NAMES.index("assistant_think")
    assert 850 <= by_name["en_technical"]["n_tokens"] <= 950 and 550 <= by_name["ko_passage"]["n_tokens"] <= 650
    for name, p in by_name.items():
        ids, roles = p["ids"], p["roles"]
        assert ids[0] == 1 and len(roles) == len(ids) == p["n_tokens"]
        if name == "chat_default":
            continue
        # a complete conversation: ... <|startofturn|><|assistant|><think>...</think>answer<|endofturn|><|endoftext|>
        assert ids[-2:] == [6, 0] and roles[-2] == ROLE_NAMES.index("assistant") and roles[-1] == 0
        last_asst = max(i for i in range(len(ids) - 1) if ids[i] == 4 and ids[i - 1] == 5)
        assert ids[last_asst + 1] == 11 and roles[last_asst + 1] == ROLE_NAMES.index("assistant_think")
        assert p["n_assistant_tokens"] >= 100
        assert all(r in (3, 4) for r in roles[last_asst + 1 : -1])
        assert tokenizer.decode(ids) == p["text"]
    chat = by_name["multi_turn_chat"]
    assert sum(1 for i in range(1, len(chat["ids"])) if chat["ids"][i - 1 : i + 1] == [5, 4]) == 3
