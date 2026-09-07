# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Host-only reference generator for ``tests/accuracy/test_teacher_forced.py`` (design 5.7 / bring-up step 7.2-5).

Loads the bf16 ``SolarOpenForCausalLM`` checkpoint on the CPU (~205 GB of weights; the whole model, never together
with another whole-model load or with a device process on a 503 GB box), renders a few KO/EN prompts through Solar's
chat template with a FIXED date (the template stamps ``strftime_now("%Y-%m-%d")`` into the system prompt, so without
this the token ids change from day to day), greedily generates ``--num-new-tokens`` tokens per prompt and saves, per
prompt, the token ids and the per-step logits: the full bf16 vector (lossless - the bf16 model emits bf16 logits),
the top-64 values / indices in fp32, and the argmax sequence. The device test replays exactly these token ids
(prompt + reference continuation) through the TT model and compares its per-step logits with this file.

Generation continues through the stop tokens {2, 24, 25} (teacher forcing does not care; the first stop position is
recorded so the test can also report the metrics up to it).

    source env.sh   # HF_MODEL, TT_CACHE_PATH, HF_HUB_OFFLINE
    timeout 10800 python models/demos/solar_open/tests/accuracy/gen_reference.py --out $TT_CACHE_PATH/teacher_forced_reference.pt

Runtime on the bring-up box (Xeon Silver 4510, 24 threads): 54 s load (shards in the page cache), 5-10 s prefill
per prompt, ~0.5 s per generated token; ~205 GB RSS.
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import torch

DEFAULT_PROMPTS_FILE = "models/demos/solar_open/demo/sample_prompts/input_data_questions_ko_en_prefill_128.json"
# 2 Korean + 2 English prompts of the KO/EN demo set: the KO capital prompt (its first token is a 0.5-logit
# <|think|> / <|content|> near-tie that the device and HF resolved differently in the demo), the KO Olympics prompt
# (a post-cutoff fact the device stuttered on), the EN largest-desert prompt (the device's greedy reasoning cycled
# without reaching <|content|>) and the EN capital-of-Australia prompt.
DEFAULT_PROMPT_INDICES = (0, 15, 22, 16)
DEFAULT_DATE = "2026-09-07"
DEFAULT_NUM_NEW_TOKENS = 64
TOP_K = 64
STOP_IDS = (2, 24, 25)  # generation_config.json eos: <|endoftext|>, <|flush|>, <|calls|>
REFERENCE_FORMAT = 1


def default_reference_path():
    """``$SOLAR_OPEN_TF_REFERENCE``, else ``<TT_CACHE_PATH or ~/.cache/tenstorrent/Solar-Open-100B>/teacher_forced_reference.pt``."""
    explicit = os.getenv("SOLAR_OPEN_TF_REFERENCE")
    if explicit:
        return Path(explicit)
    root = os.getenv("TT_CACHE_PATH") or os.path.expanduser("~/.cache/tenstorrent/Solar-Open-100B")
    return Path(root) / "teacher_forced_reference.pt"


def render_prompt_ids(tokenizer, prompt, reasoning_effort, default_system_prompt, date_string):
    """Token ids of ``prompt`` through Solar's chat template with a fixed date (same call as ``ModelArgs.encode_prompt``).

    ``strftime_now`` is a Jinja global of transformers' template environment; a same-named template kwarg shadows it,
    which pins the "The current date is ..." sentence of the provider system prompt.
    """
    chat = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        reasoning_effort=reasoning_effort,
        default_system_prompt=default_system_prompt,
        strftime_now=lambda fmt: date_string,
    )
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return [int(t) for t in ids]


def rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("nan")


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def topk_fp32(logits_bf16, k):
    values, indices = torch.topk(logits_bf16.float(), k, dim=-1)
    return values.contiguous(), indices.contiguous()


def load_model(model_path, threads):
    from transformers import AutoConfig
    from transformers.models.solar_open.modeling_solar_open import SolarOpenForCausalLM

    torch.set_num_threads(threads)
    cfg = AutoConfig.from_pretrained(model_path)
    cfg._attn_implementation = "eager"  # contract C11: the reference attention / expert paths
    cfg._experts_implementation = "eager"  # loops over the selected experts only (fast on CPU)
    t0 = time.time()
    model = SolarOpenForCausalLM.from_pretrained(model_path, config=cfg, dtype=torch.bfloat16).eval()
    load_s = time.time() - t0
    n_bf16 = sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16)
    n_other = sum(p.numel() for p in model.parameters() if p.dtype != torch.bfloat16)
    print(
        f"loaded {model_path} in {load_s:.0f} s: {n_bf16 / 1e9:.1f} B bf16 params, {n_other / 1e6:.1f} M non-bf16 params "
        f"(router biases); RSS {rss_gb():.1f} GB, peak {peak_rss_gb():.1f} GB",
        flush=True,
    )
    assert n_other < 1e6, "the reference must stay bf16 (no fp32 upcast)"
    return model, load_s


@torch.inference_mode()
def generate_reference(model, prompt_ids, num_new_tokens, top_k=TOP_K):
    """Greedy continuation of ``prompt_ids`` with the per-step logits. Step t's logits predict continuation token t."""
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    t0 = time.time()
    out = model(ids, use_cache=True)
    prefill_s = time.time() - t0
    prompt_logits = out.logits[0]  # [L, V] bf16: position i predicts prompt token i + 1 (the last one predicts step 0)
    assert prompt_logits.dtype == torch.bfloat16, prompt_logits.dtype
    prompt_top_values, prompt_top_indices = topk_fp32(prompt_logits, top_k)
    past = out.past_key_values
    step_logits = [prompt_logits[-1].clone()]
    gen = [int(torch.argmax(step_logits[-1].float()))]
    t1 = time.time()
    for _ in range(num_new_tokens - 1):
        o = model(torch.tensor([[gen[-1]]], dtype=torch.long), past_key_values=past, use_cache=True)
        past = o.past_key_values
        step_logits.append(o.logits[0, -1].clone())
        gen.append(int(torch.argmax(step_logits[-1].float())))
    decode_s = time.time() - t1
    logits = torch.stack(step_logits)  # [T, V] bf16
    top_values, top_indices = topk_fp32(logits, top_k)
    stops = [t for t, tok in enumerate(gen) if tok in STOP_IDS]
    return {
        "gen_ids": torch.tensor(gen, dtype=torch.long),
        "first_stop_pos": stops[0] if stops else -1,
        "logits": logits,
        "top_values": top_values,
        "top_indices": top_indices,
        "prompt_top_values": prompt_top_values,
        "prompt_top_indices": prompt_top_indices,
        "prefill_s": prefill_s,
        "decode_s_per_token": decode_s / max(1, num_new_tokens - 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.getenv("HF_MODEL", "upstage/Solar-Open-100B"))
    ap.add_argument("--prompts-file", default=DEFAULT_PROMPTS_FILE)
    ap.add_argument("--prompt-indices", type=int, nargs="+", default=list(DEFAULT_PROMPT_INDICES))
    ap.add_argument("--num-new-tokens", type=int, default=DEFAULT_NUM_NEW_TOKENS)
    ap.add_argument("--reasoning-effort", default=os.getenv("SOLAR_OPEN_REASONING_EFFORT", "low"))
    ap.add_argument(
        "--default-system-prompt", type=int, default=int(os.getenv("SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT", "1") == "1")
    )
    ap.add_argument("--date", default=DEFAULT_DATE, help="fixed strftime_now date stamped into the system prompt")
    ap.add_argument("--threads", type=int, default=max(8, (os.cpu_count() or 16) // 2))
    ap.add_argument("--out", type=Path, default=None, help=f"output .pt (default {default_reference_path()})")
    args = ap.parse_args(argv)
    out_path = args.out or default_reference_path()

    from transformers import AutoTokenizer

    print(f"free host memory before load: {os.popen('free -g | sed -n 2p').read().strip()}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.prompts_file) as f:
        all_prompts = [entry["prompt"] for entry in json.load(f)]
    prompts = [(i, all_prompts[i]) for i in args.prompt_indices]
    rendered = [
        render_prompt_ids(tokenizer, text, args.reasoning_effort, bool(args.default_system_prompt), args.date)
        for _, text in prompts
    ]
    for (i, text), ids in zip(prompts, rendered):
        print(f"prompt {i}: {text!r} -> {len(ids)} tokens", flush=True)
    print(
        f"template of prompt {prompts[0][0]}:\n{tokenizer.decode(rendered[0], skip_special_tokens=False)}", flush=True
    )

    model, load_s = load_model(args.model, args.threads)

    import transformers

    result = {
        "format": REFERENCE_FORMAT,
        "meta": {
            "model_path": str(args.model),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "experts_implementation": "eager",
            "reasoning_effort": args.reasoning_effort,
            "default_system_prompt": bool(args.default_system_prompt),
            "date_string": args.date,
            "prompts_file": args.prompts_file,
            "prompt_indices": list(args.prompt_indices),
            "num_new_tokens": args.num_new_tokens,
            "top_k": TOP_K,
            "stop_ids": list(STOP_IDS),
            "vocab_size": int(model.config.vocab_size),
            "threads": args.threads,
            "load_s": load_s,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "prompts": [],
    }
    for (i, text), ids in zip(prompts, rendered):
        t0 = time.time()
        ref = generate_reference(model, ids, args.num_new_tokens)
        wall = time.time() - t0
        gen_text = tokenizer.decode(ref["gen_ids"].tolist(), skip_special_tokens=False)
        margins = ref["top_values"][:, 0] - ref["top_values"][:, 1]
        print(
            f"\nPROMPT {i} ({len(ids)} tokens): {text!r}\n  prefill {ref['prefill_s']:.1f} s, "
            f"{ref['decode_s_per_token']:.2f} s/token, {wall:.0f} s total; RSS {rss_gb():.1f} GB\n"
            f"  first stop token at step {ref['first_stop_pos']}; top-1 margin min {margins.min():.3f} / "
            f"median {margins.median():.3f}; steps with margin < 0.5: {int((margins < 0.5).sum())} of {len(margins)}\n"
            f"  greedy: {gen_text!r}",
            flush=True,
        )
        ref.update({"index": i, "text": text, "prompt_ids": torch.tensor(ids, dtype=torch.long), "gen_text": gen_text})
        result["prompts"].append(ref)

    result["meta"]["peak_rss_gb"] = peak_rss_gb()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    print(
        f"\nsaved {out_path} ({out_path.stat().st_size / 2**20:.0f} MiB); peak RSS {peak_rss_gb():.1f} GB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
