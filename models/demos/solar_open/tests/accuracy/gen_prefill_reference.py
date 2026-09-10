# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Host-only HF reference for the packed-vs-sequential prefill test (``tests/unit/test_batched_prefill.py``, phase 3e / A0).

Loads the bf16 ``SolarOpenForCausalLM`` checkpoint on the CPU (~205 GB of weights: never together with a device process
or another whole-model load on the 503 GB box), renders EVERY prompt of the 128-token KO/EN demo set through Solar's
chat template exactly like ``ModelArgs.encode_prompt`` does in that test (``reasoning_effort`` = $SOLAR_OPEN_REASONING_EFFORT
or "low" -- the test imports ``demo/text_demo.py``, whose ``os.environ.setdefault`` makes "low" the effective default there,
i.e. an empty think block ``<|think|><|end|><|begin|>assistant`` closes every prompt, 78 tokens for prompt 0 instead of the
74 of ``encode_prompt``'s own "high" default; the default system prompt on; the template date pinned to the day the test's
digits were recorded -- ``conftest.RECORDED_TEMPLATE_DATE`` = 2026-09-08 -- so the ids are the ones the
``pinned_template_date`` fixture produces),
prefills each prompt unpadded and saves, per prompt, the token ids and the bf16 logits of the LAST position (the
first-token distribution both device arms of the test read back), plus their fp32 top-64. The device test loads the
file, asserts that its own ids equal the recorded ones user by user and ranks BOTH device arms (sequential per-user
prefill, packed 32 x 128 pass) against this reference: KL(HF || arm), PCC, top-1 and the HF margin per user.

    source env.sh   # HF_MODEL, TT_CACHE_PATH, HF_HUB_OFFLINE
    timeout 3600 python models/demos/solar_open/tests/accuracy/gen_prefill_reference.py   # -> $TT_CACHE_PATH/prefill_reference_128.pt

``SOLAR_OPEN_PREFILL_REFERENCE`` points the test (and ``--out`` here) at another file; the test refuses a reference whose
date, reasoning effort or ids differ from its own. Runtime on the bring-up box (48 threads): 430 s load from disk (~1 min
with the shards in the page cache), 3-5 s per prompt, ~300 GB RSS.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:  # `python path/to/gen_prefill_reference.py` puts the script dir first, not the repo
    sys.path.insert(0, str(_REPO_ROOT))

from models.demos.solar_open.tests.accuracy.gen_reference import (  # noqa: E402
    DEFAULT_PROMPTS_FILE,
    TOP_K,
    load_model,
    peak_rss_gb,
    render_prompt_ids,
    rss_gb,
    topk_fp32,
)

# == conftest.RECORDED_TEMPLATE_DATE (the day the batched-prefill floors were measured; the test asserts the match).
DEFAULT_DATE = "2026-09-08"
PREFILL_REFERENCE_ENV = "SOLAR_OPEN_PREFILL_REFERENCE"
PREFILL_REFERENCE_FORMAT = 1


def default_prefill_reference_path():
    """``$SOLAR_OPEN_PREFILL_REFERENCE``, else ``<TT_CACHE_PATH or ~/.cache/tenstorrent/Solar-Open-100B>/prefill_reference_128.pt``."""
    explicit = os.getenv(PREFILL_REFERENCE_ENV)
    if explicit:
        return Path(explicit)
    root = os.getenv("TT_CACHE_PATH") or os.path.expanduser("~/.cache/tenstorrent/Solar-Open-100B")
    return Path(root) / "prefill_reference_128.pt"


@torch.inference_mode()
def last_position_logits(model, prompt_ids):
    """bf16 logits [V] of the last prompt position (the first generated token's distribution) and the prefill wall."""
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    t0 = time.time()
    out = model(ids, use_cache=False)
    wall = time.time() - t0
    logits = out.logits[0, -1].clone()
    assert logits.dtype == torch.bfloat16, logits.dtype
    return logits, wall


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.getenv("HF_MODEL", "upstage/Solar-Open-100B"))
    ap.add_argument("--prompts-file", default=DEFAULT_PROMPTS_FILE)
    ap.add_argument("--num-prompts", type=int, default=None, help="first N prompts of the file (default: all)")
    # What the device test effectively encodes with (module docstring): text_demo's "low" default, the system prompt on.
    ap.add_argument("--reasoning-effort", default=os.getenv("SOLAR_OPEN_REASONING_EFFORT", "low"))
    ap.add_argument(
        "--default-system-prompt", type=int, default=int(os.getenv("SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT", "1") == "1")
    )
    ap.add_argument("--date", default=DEFAULT_DATE, help="fixed strftime_now date stamped into the system prompt")
    ap.add_argument("--threads", type=int, default=max(8, (os.cpu_count() or 16) - 8))
    ap.add_argument("--out", type=Path, default=None, help=f"output .pt (default {default_prefill_reference_path()})")
    args = ap.parse_args(argv)
    out_path = args.out or default_prefill_reference_path()

    import transformers
    from transformers import AutoTokenizer

    print(f"free host memory before load: {os.popen('free -g | sed -n 2p').read().strip()}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.prompts_file) as f:
        all_prompts = [entry["prompt"] for entry in json.load(f)]
    prompts = list(enumerate(all_prompts))[: args.num_prompts]
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

    result = {
        "format": PREFILL_REFERENCE_FORMAT,
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
            "num_prompts": len(prompts),
            "top_k": TOP_K,
            "vocab_size": int(model.config.vocab_size),
            "threads": args.threads,
            "load_s": load_s,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "prompts": [],
    }
    for (i, text), ids in zip(prompts, rendered):
        logits, wall = last_position_logits(model, ids)
        top_values, top_indices = topk_fp32(logits, TOP_K)
        margin = float(top_values[0] - top_values[1])
        print(
            f"PROMPT {i} ({len(ids)} tokens) prefill {wall:.1f} s; top-1 {int(top_indices[0])} "
            f"({tokenizer.decode([int(top_indices[0])])!r}) margin {margin:.3f} over {int(top_indices[1])} "
            f"({tokenizer.decode([int(top_indices[1])])!r}); RSS {rss_gb():.1f} GB",
            flush=True,
        )
        result["prompts"].append(
            {
                "index": i,
                "text": text,
                "prompt_ids": torch.tensor(ids, dtype=torch.long),
                "logits": logits,
                "top_values": top_values,
                "top_indices": top_indices,
                "prefill_s": wall,
            }
        )

    result["meta"]["peak_rss_gb"] = peak_rss_gb()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    print(
        f"\nsaved {out_path} ({out_path.stat().st_size / 2**20:.1f} MiB); peak RSS {peak_rss_gb():.1f} GB", flush=True
    )


if __name__ == "__main__":
    main()
