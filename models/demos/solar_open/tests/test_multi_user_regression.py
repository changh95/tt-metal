# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
ISL/OSL x batch multi-user regression sweep for Solar-Open-100B on single-row meshes (8x Blackhole P150, 1x8, TP=8).

One pytest case per batch size (1, 2, 4, 8, 16, 32): the model is built for that batch, then every
(input length, output length) pair of the tt-inference-server benchmark sweep
(reference_config/benchmarking/benchmark_config.py::BENCHMARK_ISL_OSL_PAIRS) that fits the per-batch context
budget is run the way the benchmark client does it: all users prefilled (sequentially, as the 1x8 mesh does it),
then exactly OSL decode steps with no EOS stopping. The context budget mirrors the server's concurrency cap
(``max_tokens_all_users // (isl + osl)``): a batch of B users gets ``min(64K, 512K // B)`` tokens of context each
(a 512K-token pool is 6.4 GiB of bfp8 KV per device at 13,056 B/token, inside the 8 GiB default budget of
tt/common.py), so long inputs are swept at small batch sizes only.

Per pair the sweep records the prefill time (TTFT of the first / mean / last user), decode step statistics
(mean / p50 / p99, first traced step separately), per-user and aggregate tokens/s and the board temperatures around
the prefill, and checks the outputs:
  * every generated token id is a valid vocab id and the text decodes;
  * the first generated token of every user is ``<|think|>`` (22) or ``<|content|>`` (23): Solar's chat template
    ends the prompt with ``<|begin|>assistant``, so garbage prefill logits fail this gate even when the decoded
    garbage is diverse enough to slip past the degeneracy heuristic;
  * outputs are not degenerate (no long runs of one token, reasonable token diversity);
  * for the 128-token QA prompts (16 Korean + 16 English questions) the answer keyword appears somewhere in the
    stop-truncated generation (think block + answer: even under ``reasoning_effort=low`` the model opens a
    ``<|think|>`` block for most prompts and states the answer inside it) for >= 50 % of the users; the number of
    users that reached ``<|content|>`` within the OSL budget is recorded too (think-only outputs at OSL 128 can
    legitimately miss the keyword, real garbage fails the gate);
  * for the long prompts (the same Gutenberg excerpt for every user) the users' outputs are compared with each
    other, which doubles as a cross-user isolation signal (reported, not asserted: greedy decode is not
    bit-reproducible on device and near-tie tokens can flip).

Results are appended to ``generated/solar_open_multi_user_regression/<model>_<mesh><tag>.jsonl`` (one row per
pair, schema shared with the GPT-OSS sweep so ``tests/sweep/report.py`` renders both) and printed as a Markdown
table at the end of each case.

Solar specifics: the model is built through ``tt/common.py::create_tt_model`` (the demo's path) rather than
``demo/text_demo.py``, whose import chain opens the devices at collection time; prompts always go through Solar's
chat template (``ModelArgs.encode_prompt``, ``reasoning_effort`` from ``SOLAR_OPEN_REASONING_EFFORT``, default
"low" here as in the demo; the long-prompt files contribute their Gutenberg context alone as the user message, the
way the demo's ``prefill_1k``..``prefill_32k`` cases and the GPT-OSS sweep load them); the stop set is
``ModelArgs.stop_token_ids`` ({2, 24, 25}); ``warmup_prefill=False`` everywhere (the generic warm-up captures its
traces in an order that corrupts prefill on this box, tt-metal #52176) with an explicit eager pre-compile of every
prefill length before the first trace is captured; the decode batch equals ``max_local_batch_size`` (single row),
and every batch of {1, 2, 4, 8, 16, 32} maps onto the Blackhole user grids of ``tt/attention/config.py``
(<= 8 users and multiples of 32 on the 8x8 grid, 16 on the 13x10 device grid, concat grid 8x2).

    source env.sh   # HF_MODEL=/path/to/Solar-Open-100B, TT_CACHE_PATH (warm cache), MESH_DEVICE=P150x8
    pytest models/demos/solar_open/tests/test_multi_user_regression.py -k 1x8                  # all batch sizes
    pytest "models/demos/solar_open/tests/test_multi_user_regression.py::test_multi_user_regression[blackhole-1x8-batch32]"
    SOLAR_OPEN_REGRESSION_PAIRS="128:128,1024:128" pytest ... -k "1x8 and batch1"               # subset of pairs
    SOLAR_OPEN_NUM_DEVICES=8 pytest models/demos/solar_open/tests/test_multi_user_regression.py --collect-only -q  # host

Environment knobs (all optional): SOLAR_OPEN_REGRESSION_TAG (suffix of the results file), _KV_TOKENS (total KV
tokens per mesh, default 512K), _POW2_CONTEXT (1 = round the per-user context down to a power of two, the server rule;
0 = round to a block multiple so e.g. 1,056,000 tokens // 32 users hosts the 32768/128 pair), _PAIRS ("isl:osl,..."),
_COOLDOWN_C / _COOLDOWN_TIMEOUT_S / _FULL_AICLK_MHZ (thermal gate,
see below), _PAGE_TABLE_SEED (default 1234), _DECODE_TRACE (0 = eager decode, debug), _TIMEOUT_S (pytest-timeout limit
per batch case, default 7200; the marker overrides a command-line --timeout, so the driver sets this instead).
"""

import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams
from models.demos.solar_open.config import MoEOptions
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.solar_open.tt.common import check_kv_budget, create_tt_model
from models.tt_transformers.tt.common import PagedAttentionConfig, get_padded_prefill_len, preprocess_inputs_prefill
from models.tt_transformers.tt.generator import Generator

# Same template default as demo/text_demo.py: answer without a mandatory think block. ModelArgs.encode_prompt reads
# the variable at call time, so an exported value wins over this default.
os.environ.setdefault("SOLAR_OPEN_REASONING_EFFORT", "low")

# tt-inference-server BENCHMARK_ISL_OSL_PAIRS, minus the points this box cannot host (>= 64K per user on a
# single-row mesh, the demo's 64K prefill cap) and the 10000-token point (no matching sample prompt).
ISL_OSL_PAIRS = [
    (128, 128),
    (128, 1024),
    (1024, 128),
    (2048, 128),
    (4096, 128),
    (8192, 128),
    (8192, 1024),
    (16384, 128),
    (32768, 128),
]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
# Total KV tokens the sweep may allocate per mesh: 512K tokens = 8192 blocks of 64 = 6.4 GiB of bfp8 KV per device
# (48 full-attention layers, 1 KV head per device at TP=8, 13,056 B/token), inside the 8 GiB default budget of
# tt/common.py. Mirrors the server's context-capped concurrency: a batch of B gets min(64K, 512K // B) tokens each.
TOTAL_KV_TOKENS = int(os.getenv("SOLAR_OPEN_REGRESSION_KV_TOKENS", 512 * 1024))
MAX_CONTEXT_PER_USER = 64 * 1024
BLOCK_SIZE = 64
QA_MIN_ACCURACY = 0.5

SAMPLE_PROMPTS = "models/tt_transformers/demo/sample_prompts"
PROMPT_FILES = {
    128: "models/demos/solar_open/demo/sample_prompts/input_data_questions_ko_en_prefill_128.json",
    1024: f"{SAMPLE_PROMPTS}/input_data_long_1k.json",
    2048: f"{SAMPLE_PROMPTS}/input_data_long_2k.json",
    4096: f"{SAMPLE_PROMPTS}/input_data_long_4k.json",
    8192: f"{SAMPLE_PROMPTS}/input_data_long_8k.json",
    16384: f"{SAMPLE_PROMPTS}/input_data_long_16k.json",
    32768: f"{SAMPLE_PROMPTS}/input_data_long_32k.json",
}
# Where models/tt_transformers/demo/simple_text_demo.py::load_and_cache_context keeps the downloaded Gutenberg texts
# (file name = md5 of the URL); reused here so the sweep needs no network once the demo's long cases have run.
CONTEXT_CACHE_DIR = Path("models/tt_transformers/demo/context_cache")

# Expected answer keywords for the 32 KO/EN QA prompts, in file order (any keyword matching counts; the model may
# reason in either language, so Korean prompts also accept the English answer). Matching rules: numbers must not
# touch other digits, ASCII words need letter/digit boundaries, Korean keywords are substrings (see _keyword_present).
QA_KEYWORDS = [
    ["서울", "seoul"],  # 대한민국의 수도
    ["100"],  # 물의 끓는점 (섭씨)
    ["세종", "sejong"],  # 한글을 만든 왕
    ["목성", "jupiter"],  # 태양계에서 가장 큰 행성
    ["365", "366"],  # 일 년은 며칠
    ["에베레스트", "everest"],  # 지구에서 가장 높은 산
    ["180"],  # 삼각형 내각의 합
    ["이산화탄소", "carbon dioxide", "co2", "co₂"],  # 광합성에 필요한 기체
    ["1592"],  # 임진왜란
    ["태평양", "pacific"],  # 가장 넓은 바다
    ["4", "네 개", "넷", "four"],  # 심장의 방
    ["144"],  # 12 x 12
    ["엔", "yen", "¥", "円"],  # 일본의 화폐 단위
    ["300,000", "300000", "299,792", "299792", "30만", "3×10^5", "3x10^5", "300 000"],  # 빛의 속도 (km/s)
    ["박경리", "kyung-ni", "kyongni", "kyung-ri", "kyeong-ri", "kyeongni"],  # 소설 '토지'의 작가
    ["4", "four", "2", "1896"],  # 올림픽 주기 (4년; 2년 = 하계/동계 교대, 1896 = 첫 대회)
    ["canberra"],  # capital of Australia
    ["seven", "7"],  # continents
    ["shakespeare"],  # Romeo and Juliet
    ["au"],  # chemical symbol for gold
    ["1,440", "1440"],  # minutes in a day
    ["mars"],  # Red Planet
    ["antarctic", "antarctica", "sahara"],  # largest desert (both defensible answers)
    ["1945"],  # end of World War II
    ["12", "twelve"],  # sqrt(144)
    ["heart"],  # organ that pumps blood
    ["diamond", "diamonds"],  # hardest natural substance
    ["einstein"],  # general relativity
    ["four", "4"],  # violin strings
    ["pound", "pounds", "sterling", "gbp"],  # currency of the United Kingdom
    ["nitrogen", "n2", "n₂"],  # most of the atmosphere
    ["206"],  # bones in the adult human body
]

# Special tokens of the Solar chat template (tokenizer_config.json added_tokens_decoder); ids are looked up through
# the tokenizer at run time and these values are the fallback.
THINK_TOKEN, THINK_TOKEN_ID = "<|think|>", 22
CONTENT_TOKEN, CONTENT_TOKEN_ID = "<|content|>", 23

# Debug knob: SOLAR_OPEN_REGRESSION_DECODE_TRACE=0 runs decode eagerly (slower, but isolates trace-replay issues).
DECODE_TRACE = os.getenv("SOLAR_OPEN_REGRESSION_DECODE_TRACE", "1") not in ("0", "false", "False")

# Thermal guard. On this P150x8 box board 1 is the hottest; during sustained prefill it reaches the throttle point
# (>84-87 C, AI clock 1350 -> 800 MHz) and a traced decode step launched while a board is deep-throttled has
# deadlocked inside paged SDPA decode (GPT-OSS sweep on the same box: writer stuck in a NOC write barrier, the other
# devices waiting in the next CCL). Set SOLAR_OPEN_REGRESSION_COOLDOWN_C=<max asic temperature> to wait (up to
# COOLDOWN_TIMEOUT_S) for every board to cool below it before each timed prefill and before the first decode step.
# With the mesh open the fabric routers and dispatch cores spin at full clock (64-87 W per board at idle), so the
# hottest board settles at ~82-86 C: a gate below that is met only while the box is still warming up, and the wait
# returns early once the boards have stopped cooling (plateau) as long as no board is clock-throttled.
COOLDOWN_C = float(os.getenv("SOLAR_OPEN_REGRESSION_COOLDOWN_C", "0"))
COOLDOWN_TIMEOUT_S = float(os.getenv("SOLAR_OPEN_REGRESSION_COOLDOWN_TIMEOUT_S", "180"))
FULL_AICLK_MHZ = int(os.getenv("SOLAR_OPEN_REGRESSION_FULL_AICLK_MHZ", "1350"))  # P150 AI clock when not throttled
COOLDOWN_PLATEAU_SAMPLES = 6  # ~35-40 s of tt-smi samples without a 0.5 C drop = the boards have stopped cooling

_PROMPT_CACHE = {}
_NUMERIC_RE = re.compile(r"^[\d.,]+$")


def _special_token_id(tokenizer, token, fallback):
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
    except Exception:  # pragma: no cover - tokenizer without the token
        tid = None
    unk = getattr(tokenizer, "unk_token_id", None)
    return int(tid) if isinstance(tid, int) and tid >= 0 and tid != unk else fallback


def _first_token_ids(tokenizer):
    """Every Solar reply opens with <|think|> (reasoning) or <|content|> (direct answer)."""
    return {
        _special_token_id(tokenizer, THINK_TOKEN, THINK_TOKEN_ID),
        _special_token_id(tokenizer, CONTENT_TOKEN, CONTENT_TOKEN_ID),
    }


def _truncate_at_stop(tokens, stop_ids):
    """Tokens up to (excluding) the first end-of-generation token; the sweep keeps decoding past it for timing."""
    for i, t in enumerate(tokens):
        if t in stop_ids:
            return tokens[:i]
    return tokens


def _keyword_present(text, keyword):
    """Keyword match on lower-cased text.

    Numbers must not touch other digits ('7' does not match '49' or '2017', '4' does not match '144' or '1,440');
    ASCII words need letter/digit boundaries ('au' does not match 'australia'); Korean keywords match as substrings
    because particles attach directly ('서울입니다', '100도', '세종대왕')."""
    kw = keyword.lower()
    if _NUMERIC_RE.match(kw):
        pattern = r"(?<!\d)(?<!\d[.,])" + re.escape(kw) + r"(?!\d)(?![.,]\d)"
    elif kw.isascii():
        pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
    else:
        return kw in text
    return re.search(pattern, text) is not None


def _load_context(url, max_length=None):
    """The tt_transformers demo's Gutenberg context: read from its cache, else download and cache; clipped to
    max_length characters (as load_and_cache_context does)."""
    CONTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CONTEXT_CACHE_DIR / hashlib.md5(url.encode()).hexdigest()
    if cache_file.exists():
        text = cache_file.read_text(encoding="utf-8")
    else:
        import requests

        response = requests.get(url, timeout=120)
        assert (
            response.status_code == 200
        ), f"could not fetch the long-prompt context {url}: HTTP {response.status_code}"
        text = response.text
        cache_file.write_text(text, encoding="utf-8")
        logger.info(f"downloaded and cached the long-prompt context {url}")
    return text[:max_length] if max_length else text


def _load_prompt_file(path):
    """Prompt strings of a sample_prompts json file. Entries with a "context" contribute the (clipped) context alone,
    the tt_transformers loader's instruct=False rule that the demo's prefill_1k..32k cases and the GPT-OSS sweep use;
    the chat template is applied later by ModelArgs.encode_prompt."""
    with open(path) as f:
        entries = json.load(f)
    prompts = []
    for entry in entries:
        if "context" in entry:
            prompts.append(_load_context(entry["context"], entry.get("max_length")))
        else:
            prompts.append(entry["prompt"])
    assert prompts, f"no prompts in {path}"
    return prompts


def _prompts_for(isl, batch):
    """Batch prompts for a nominal input length: the 128 case cycles the 32 distinct KO/EN QA prompts (a batch above
    32 repeats them), the long files hold one prompt that every user gets."""
    if isl not in _PROMPT_CACHE:
        _PROMPT_CACHE[isl] = _load_prompt_file(PROMPT_FILES[isl])
    prompts = _PROMPT_CACHE[isl]
    return [prompts[i % len(prompts)] for i in range(batch)]


def _seeded_page_table(batch, max_num_blocks, seed):
    """The demo's page table (simple_text_demo.create_tt_page_table: a random block permutation) from an explicit
    seed, so a layout-dependent failure reproduces from run to run (the seed is recorded in the results)."""
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(max_num_blocks, generator=generator)
    return torch.argsort(permutation).reshape(batch, max_num_blocks // batch)


def _selected_pairs():
    env = os.getenv("SOLAR_OPEN_REGRESSION_PAIRS")
    if not env:
        return ISL_OSL_PAIRS
    return [tuple(int(x) for x in p.split(":")) for p in env.split(",") if p.strip()]


def _clear_kv_caches(models):
    for m in models:
        for layer in m.layers:
            k_cache, v_cache = layer.self_attn.layer_past
            ttnn.mul(k_cache, 0, output_tensor=k_cache)
            ttnn.mul(v_cache, 0, output_tensor=v_cache)


def _degenerate(tokens, max_run=48, min_unique_ratio=0.15):
    """Heuristic garbage detector: long runs of one token or almost no token diversity."""
    if len(tokens) < 32:
        return False
    run = best = 1
    for a, b in zip(tokens, tokens[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    unique_ratio = len(set(tokens)) / len(tokens)
    return best > max_run or unique_ratio < min_unique_ratio


def _percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[idx]


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - best effort metadata
        return "unknown"


def _board_telemetry():
    """Returns (max_temp_C, min_aiclk_MHz) across boards via tt-smi, or (None, None) if unavailable."""
    try:
        out = subprocess.run(["tt-smi", "-s", "--snapshot_no_tty"], capture_output=True, text=True, timeout=60).stdout
        data = json.loads(out[out.find("{") :])
        temps, clocks = [], []
        for dev in data.get("device_info", []):
            tel = dev.get("telemetry", {})
            temps.append(float(str(tel.get("asic_temperature", "nan")).strip()))
            clocks.append(int(str(tel.get("aiclk", "0")).strip()))
        return (max(temps) if temps else None, min(clocks) if clocks else None)
    except Exception as e:  # pragma: no cover - telemetry is best effort
        logger.warning(f"tt-smi telemetry unavailable: {e}")
        return None, None


def _wait_for_cooldown(stage):
    """Blocks until every board is below COOLDOWN_C and none is clock-throttled, or the boards have stopped cooling
    (open-mesh idle equilibrium, see above), or the timeout passes. Returns the last max temperature."""
    if COOLDOWN_C <= 0:
        return None
    t0 = time.perf_counter()
    max_t, clk = _board_telemetry()
    throttled = clk is not None and 0 < clk < FULL_AICLK_MHZ
    history = [max_t] if max_t is not None else []
    if max_t is not None and (max_t > COOLDOWN_C or throttled):
        logger.info(
            f"   cooldown before {stage}: waiting, max board temperature {max_t} C > {COOLDOWN_C} C"
            + (f", min AI clock {clk} MHz (throttled)" if throttled else "")
        )
    while max_t is not None and (max_t > COOLDOWN_C or throttled) and time.perf_counter() - t0 < COOLDOWN_TIMEOUT_S:
        n = COOLDOWN_PLATEAU_SAMPLES
        if not throttled and len(history) > n and min(history[-n:]) > min(history[:-n]) - 0.5:
            logger.info(
                f"   cooldown before {stage}: boards stopped cooling at {max_t} C after "
                f"{time.perf_counter() - t0:.0f}s (open-mesh idle equilibrium, AI clock {clk} MHz), continuing"
            )
            break
        time.sleep(5)
        max_t, clk = _board_telemetry()
        throttled = clk is not None and 0 < clk < FULL_AICLK_MHZ
        if max_t is not None:
            history.append(max_t)
    waited = time.perf_counter() - t0
    if waited > 6:
        logger.info(
            f"   cooldown before {stage}: waited {waited:.0f}s, max board temperature now {max_t} C, min AI clock {clk} MHz"
        )
    return max_t


def _prefill(generator, models, mesh_device, tt_kv_cache, page_table, input_tokens, decoding_pos, enable_trace=True):
    """One timed prefill of all users on a cleared KV cache. warmup_prefill=False is the Solar path (as in
    text_demo): it skips the generic batch-1 warm-up sweep and hoists the decode-trace allocations ahead of the
    first trace capture (see Generator.prefill_forward_text and tt-metal #52176)."""
    _clear_kv_caches(models)
    generator.prev_page_table = None
    ttnn.synchronize_device(mesh_device)  # keep the (async) cache clear out of the prefill timing
    t0 = time.perf_counter()
    logits = generator.prefill_forward_text(
        input_tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=enable_trace,
        warmup_prefill=False,
    )
    ttnn.synchronize_device(mesh_device)
    return logits, time.perf_counter() - t0


def _encode_batch(prompts, tokenizer, model_args, osl, max_seq_len):
    """Chat-templated, right-padded prefill tokens [B, padded] plus each user's real prompt length."""
    input_tokens, _encoded, decoding_pos, _prefill_lens = preprocess_inputs_prefill(
        prompts, tokenizer, model_args, instruct=False, max_generated_tokens=osl, max_prefill_len=max_seq_len
    )
    return torch.stack(input_tokens).view(len(prompts), -1), decoding_pos


def _precompile_prefill_lengths(
    generator, models, mesh_device, model_args, tt_kv_cache, page_table, tokenizer, isls, max_seq_len
):
    """Compile every prefill length (one user, eager, no trace) BEFORE any trace is captured.

    Programs and persistent tensors (e.g. the cached RoPE slices) created while a trace is live may be placed in a
    trace's freed intermediate address range and be overwritten by a later replay -- tt-metal's
    TT_METAL_TRACE_ALLOC_TRACKING flags exactly this, and the symptom is a garbage prefill or a device hang some
    pairs later (seen on this box in the GPT-OSS sweep). tt_transformers avoids it with warmup_model_prefill, which
    this model disables (#52176), so the sweep does its own eager warm-up up front. Returns {padded_len: seconds}."""
    compile_times = {}
    for isl in sorted(isls):
        input_tokens, decoding_pos = _encode_batch(_prompts_for(isl, 1), tokenizer, model_args, 1, max_seq_len)
        padded_len = get_padded_prefill_len(int(max(decoding_pos)))
        if padded_len in compile_times:
            continue
        _, compile_times[padded_len] = _prefill(
            generator,
            models,
            mesh_device,
            tt_kv_cache,
            page_table[:1],
            input_tokens,
            decoding_pos[:1],
            enable_trace=False,
        )
        logger.info(f"   pre-compiled prefill length {padded_len} in {compile_times[padded_len]:.1f}s (eager, 1 user)")
    return compile_times


def _run_pair(
    generator,
    models,
    mesh_device,
    model_args,
    tt_kv_cache,
    page_table,
    tokenizer,
    prompts,
    isl,
    osl,
    sampling,
    max_seq_len,
    warmed_lengths,
    compile_times=None,
):
    """Prefill all users then decode exactly `osl` tokens (no EOS stop). Returns (metrics dict, token lists).

    The first time a padded prefill length is seen in a batch case, an untimed pass absorbs program compilation
    (and, for the very first pair, the decode-trace and prefill-trace capture) so that the reported prefill time is
    the steady-state number the demo also reports."""
    batch = len(prompts)
    input_tokens, decoding_pos = _encode_batch(prompts, tokenizer, model_args, osl, max_seq_len)
    encoded_len = int(max(decoding_pos))
    padded_len = get_padded_prefill_len(encoded_len)

    compile_time = compile_times.get(padded_len) if compile_times else None
    warm_time = None
    if padded_len not in warmed_lengths:
        # Program compilation is per shape, so one user is enough to warm a new padded length. The very first pair
        # still warms the whole batch: that call also prepares the decode trace, whose persistent page table must
        # have the full batch shape.
        warm_users = batch if not warmed_lengths else 1
        _, warm_time = _prefill(
            generator,
            models,
            mesh_device,
            tt_kv_cache,
            page_table[:warm_users],
            input_tokens[:warm_users],
            decoding_pos[:warm_users],
        )
        warmed_lengths.add(padded_len)
        logger.info(f"   warm pass for prefill length {padded_len} ({warm_users} users): {warm_time:.1f}s")
    temp_before_prefill = _wait_for_cooldown("prefill")
    logits, prefill_time = _prefill(generator, models, mesh_device, tt_kv_cache, page_table, input_tokens, decoding_pos)
    temp_after_prefill, clk_after_prefill = _board_telemetry() if COOLDOWN_C > 0 else (None, None)
    temp_before_decode = _wait_for_cooldown("decode")

    out_tok = torch.argmax(logits, dim=-1)  # [B, 1]
    outputs = [[int(out_tok[b])] for b in range(batch)]
    current_pos = torch.tensor(decoding_pos)

    step_times = []
    for _ in range(osl - 1):
        t1 = time.perf_counter()
        out_tok, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=DECODE_TRACE,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=sampling,
        )
        step_times.append(time.perf_counter() - t1)
        if len(step_times) == 1:
            logger.info(f"   first decode step (pos {int(current_pos.max())}) done in {1000 * step_times[0]:.1f} ms")
        current_pos += 1
        for b in range(batch):
            outputs[b].append(int(out_tok[b]))

    steady = step_times[1:] if len(step_times) > 1 else step_times
    mean_step = statistics.fmean(steady) if steady else float("nan")
    metrics = {
        "batch": batch,
        "isl_nominal": isl,
        "isl_encoded": encoded_len,
        "isl_padded": padded_len,
        "osl": osl,
        "prefill_compile_s": round(compile_time, 2) if compile_time is not None else None,
        "prefill_warm_s": round(warm_time, 2) if warm_time is not None else None,
        "prefill_total_s": round(prefill_time, 4),
        "prefill_per_user_ms": round(1000 * prefill_time / batch, 1),
        # TTFT: prefill is sequential per user on a single-row mesh and the first token is the argmax of each user's
        # prefill logits, so user k (1-based) sees k * per-user prefill: first user, mean over users, last user.
        "ttft_first_user_ms": round(1000 * prefill_time / batch, 1),
        "ttft_mean_user_ms": round(1000 * prefill_time / batch * (batch + 1) / 2, 1),
        "ttft_last_user_ms": round(1000 * prefill_time, 1),
        "first_decode_step_ms": round(1000 * step_times[0], 2) if step_times else None,
        "decode_step_mean_ms": round(1000 * mean_step, 2),
        "decode_step_p50_ms": round(1000 * _percentile(steady, 50), 2),
        "decode_step_p99_ms": round(1000 * _percentile(steady, 99), 2),
        "tok_s_user": round(1 / mean_step, 2) if steady else None,
        "tok_s_aggregate": round(batch / mean_step, 1) if steady else None,
        "e2e_s": round(prefill_time + sum(step_times), 2),
        "board_temp_before_prefill_c": temp_before_prefill,
        "board_temp_after_prefill_c": temp_after_prefill,
        "min_aiclk_after_prefill_mhz": clk_after_prefill,
        "board_temp_before_decode_c": temp_before_decode,
    }
    return metrics, outputs


def _check_outputs(tokenizer, vocab_size, isl, outputs, stop_ids, first_ids, content_id):
    """Correctness gates; returns (list of failures, info dict)."""
    failures, info = [], {}
    batch = len(outputs)
    for b, toks in enumerate(outputs):
        bad = [t for t in toks if not (0 <= t < vocab_size)]
        if bad:
            failures.append(f"user {b}: {len(bad)} token ids outside the vocabulary (e.g. {bad[:3]})")
    # Quality gates look at the generation up to the first stop token ({2, 24, 25}): the sweep keeps decoding past
    # it for timing, and whatever follows is outside the model's training distribution. Special tokens are kept in
    # the decoded text so the <|think|> / <|content|> structure stays visible in the recorded samples.
    generated = [_truncate_at_stop([t for t in toks if 0 <= t < vocab_size], stop_ids) for toks in outputs]
    texts = [tokenizer.decode(toks, skip_special_tokens=False) for toks in generated]
    info["finished_users"] = sum(1 for g, toks in zip(generated, outputs) if len(g) < len(toks))
    # Every Solar reply opens with <|think|> or <|content|> (the template ends the prompt with <|begin|>assistant),
    # so the first generated token (argmax of the prefill logits) must be one of them. Garbage prefill logits fail
    # this gate even when the decoded garbage is diverse enough to slip past the run-length / diversity heuristic.
    bad_start = [b for b, toks in enumerate(outputs) if not toks or toks[0] not in first_ids]
    info["users_bad_first_token"] = bad_start
    info["first_token_think"] = sum(1 for toks in outputs if toks and toks[0] != content_id and toks[0] in first_ids)
    info["first_token_content"] = sum(1 for toks in outputs if toks and toks[0] == content_id)
    if bad_start:
        failures.append(
            f"{len(bad_start)}/{batch} users did not start with {THINK_TOKEN} / {CONTENT_TOKEN} (prefill logits wrong): "
            + "; ".join(
                f"user {b}: token {outputs[b][0] if outputs[b] else None}, {texts[b][:80]!r}" for b in bad_start[:4]
            )
        )
    # How many users got past their think block within the OSL budget (informational: the keyword gate below reads
    # the whole generation, but an answer-less think block at OSL 128 is the expected miss, not garbage).
    info["content_reached"] = sum(1 for toks in generated if content_id in toks)
    degenerate = [b for b, toks in enumerate(generated) if _degenerate(toks)]
    info["degenerate_users"] = degenerate
    if len(degenerate) > max(1, batch // 4):
        failures.append(f"{len(degenerate)}/{batch} users produced degenerate output (users {degenerate[:8]})")

    if isl == 128:
        hits, answer_hits, misses = 0, 0, []
        for b, text in enumerate(texts):
            keywords = QA_KEYWORDS[b % len(QA_KEYWORDS)]
            lowered = text.lower()
            if any(_keyword_present(lowered, k) for k in keywords):
                hits += 1
            else:
                misses.append(b)
            answer = lowered.partition(CONTENT_TOKEN.lower())[2]
            if answer and any(_keyword_present(answer, k) for k in keywords):
                answer_hits += 1
        info["qa_accuracy"] = round(hits / batch, 3)  # keyword anywhere in the stop-truncated generation
        info["qa_answer_accuracy"] = round(answer_hits / batch, 3)  # keyword inside the <|content|> answer
        info["qa_misses"] = misses
        if hits / batch < QA_MIN_ACCURACY:
            failures.append(
                f"QA keyword accuracy {hits}/{batch} below {QA_MIN_ACCURACY:.0%}; misses: "
                + "; ".join(f"user {b}: {texts[b][:120]!r}" for b in misses[:4])
            )
    else:
        # Same prompt for every user: how many users agree with user 0 over the first 16 tokens.
        head = 16
        agree = sum(1 for toks in generated if toks[:head] == generated[0][:head])
        info["users_agreeing_with_user0_first16"] = f"{agree}/{batch}"
    info["sample_output"] = texts[0][:160]
    # The opening of every user's generation: with one prompt for all users (long files) this shows whether the
    # users that disagree with user 0 (think / content near-tie on a raw text) still produce coherent text.
    info["user_heads"] = [t[:120] for t in texts]
    # Keep enough for a post-mortem: user 0's whole generation (stop-truncated) plus every flagged user's.
    flagged = sorted(set([0] + degenerate + bad_start + (info.get("qa_misses") or [])))
    info["full_outputs"] = {str(b): texts[b] for b in flagged if b < batch}
    return failures, info


def _markdown_table(rows):
    cols = [
        ("batch", "B"),
        ("isl_nominal", "ISL"),
        ("isl_encoded", "enc"),
        ("isl_padded", "ISL pad"),
        ("osl", "OSL"),
        ("prefill_compile_s", "compile s"),
        ("prefill_total_s", "prefill s"),
        ("ttft_first_user_ms", "TTFT first ms"),
        ("ttft_mean_user_ms", "TTFT mean ms"),
        ("decode_step_mean_ms", "step ms"),
        ("decode_step_p99_ms", "p99 ms"),
        ("tok_s_user", "tok/s/user"),
        ("tok_s_aggregate", "tok/s agg"),
        ("qa_accuracy", "QA acc"),
        ("content_reached", "content"),
        ("status", "status"),
    ]
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k, _ in cols) + " |")
    return "\n".join(out)


# pytest-timeout's marker overrides a command-line --timeout, so the per-case limit is an env knob the sweep driver
# sets (SOLAR_OPEN_REGRESSION_TIMEOUT_S, default 7200 s); with --timeout-method thread a hang dumps every stack.
@pytest.mark.timeout(int(os.getenv("SOLAR_OPEN_REGRESSION_TIMEOUT_S", "7200")))
@pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=[f"batch{b}" for b in BATCH_SIZES])
@parametrize_mesh_with_fabric([(1, 8)])
def test_multi_user_regression(mesh_device, device_params, batch_size, state_dict):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] != 1 or mesh_shape[1] < 8:
        pytest.skip(f"multi-user single-row sweep targets 1x8 meshes, got {mesh_shape}")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    # Per-user context budget, power of two, at most 64K (mirrors the server's context-capped concurrency).
    max_seq_len = min(MAX_CONTEXT_PER_USER, TOTAL_KV_TOKENS // batch_size)
    if os.getenv("SOLAR_OPEN_REGRESSION_POW2_CONTEXT", "1") == "1":
        max_seq_len = 1 << (max_seq_len.bit_length() - 1)
    else:
        # SOLAR_OPEN_REGRESSION_POW2_CONTEXT=0: round down to a block multiple instead, so a raised
        # SOLAR_OPEN_REGRESSION_KV_TOKENS can host a pair whose ISL+OSL just exceeds a power of two
        # (e.g. 1,056,000 tokens // 32 users = 33,000 -> 32,960 positions for the 32768/128 pair; the
        # power-of-two rule would fall back to 32,768 and skip it, and 64K x 32 users does not fit DRAM).
        max_seq_len -= max_seq_len % BLOCK_SIZE
    paged_attention_config = PagedAttentionConfig(
        block_size=BLOCK_SIZE, max_num_blocks=batch_size * (max_seq_len // BLOCK_SIZE)
    )
    # KV budget pre-check on the config alone (nothing loaded yet): the pool must fit SOLAR_OPEN_KV_BUDGET_GIB
    # (tt/common.py; 8 GiB default with bfp8 experts admits the 512K-token default of this sweep).
    try:
        check_kv_budget(
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
            n_layers=config.num_hidden_layers,
            paged_attention_config=paged_attention_config,
            tensor_parallel=mesh_shape[1],
            moe_options=MoEOptions.from_env(),
        )
    except ValueError as e:
        pytest.skip(f"KV pool over the per-device budget for batch {batch_size}: {e}")
    pairs = [(i, o) for i, o in _selected_pairs() if i + o <= max_seq_len and i in PROMPT_FILES]
    skipped = [(i, o) for i, o in _selected_pairs() if (i, o) not in pairs]
    logger.info(
        f"batch {batch_size}: context {max_seq_len} tokens/user ({paged_attention_config.max_num_blocks} blocks of "
        f"{BLOCK_SIZE}), pairs {pairs}, skipped {skipped}; chat template reasoning_effort="
        f"{os.environ['SOLAR_OPEN_REASONING_EFFORT']}, default_system_prompt="
        f"{os.environ.get('SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT', '1') == '1'}; decode trace {DECODE_TRACE}"
    )

    # The demo's page table is an unseeded random block permutation; seed it so a layout-dependent failure
    # reproduces from run to run (the seed is recorded in the results).
    page_table_seed = int(os.getenv("SOLAR_OPEN_REGRESSION_PAGE_TABLE_SEED", "1234"))
    torch.manual_seed(page_table_seed)
    page_table = _seeded_page_table(batch_size, paged_attention_config.max_num_blocks, page_table_seed)
    # The demo's model path (prepare_solar_open_generator_args -> create_tt_model, one submesh): warm-cache build,
    # paged bfp8 KV, users on one row.
    model_args_0, model_0, tt_kv_cache_0, _state_dict = create_tt_model(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        optimizations=None,
        paged_attention_config=paged_attention_config,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        mesh_config=setup["mesh_config"],
        users_row_sharded=False,
    )
    model_args, models, tt_kv_cache = [model_args_0], [model_0], [tt_kv_cache_0]
    tokenizer = model_args_0.tokenizer
    assert model_args_0.max_local_batch_size == batch_size, "single-row mesh: the decode batch is the whole batch"
    generator = Generator(models, model_args, mesh_device, processor=None, tokenizer=tokenizer)
    assert all(getattr(m, "sampling", None) is not None for m in models), "on-device sampling expected on 1x8"
    sampling = SamplingParams(
        temperature=[0.0] * batch_size,
        top_k=[1] * batch_size,
        top_p=[1.0] * batch_size,
        enable_log_probs=[False] * batch_size,
        num_logprobs=[0] * batch_size,
    )
    vocab_size = model_args_0.vocab_size
    model_name = model_args_0.model_name
    # Generation stop set {2 <|endoftext|>, 24 <|flush|>, 25 <|calls|>} from generation_config.json; <|end|> (21)
    # closes a message but is NOT a stop token.
    stop_ids = set(model_args_0.stop_token_ids) or {tokenizer.eos_token_id}
    first_ids = _first_token_ids(tokenizer)
    content_id = _special_token_id(tokenizer, CONTENT_TOKEN, CONTENT_TOKEN_ID)
    logger.info(f"stop token ids {sorted(stop_ids)}, first-token gate {sorted(first_ids)}, content token {content_id}")

    out_dir = Path("generated/solar_open_multi_user_regression")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = os.getenv("SOLAR_OPEN_REGRESSION_TAG", "")  # e.g. "_baseline" to keep control runs in their own file
    out_file = out_dir / f"{model_name}_{mesh_shape[0]}x{mesh_shape[1]}{tag}.jsonl"
    meta = {
        "model": model_name,
        "mesh": f"{mesh_shape[0]}x{mesh_shape[1]}",
        "git": _git_rev(),
        "tag": tag,
        "time": datetime.now().isoformat(),
        "page_table_seed": page_table_seed,
        "context_per_user": max_seq_len,
        "kv_tokens_total": TOTAL_KV_TOKENS,
        "reasoning_effort": os.environ["SOLAR_OPEN_REASONING_EFFORT"],
        "expert_dtype": model_args_0.moe_options.expert_dtype_str,
        "decode_trace": DECODE_TRACE,
        "cooldown_c": COOLDOWN_C or None,
    }

    # Eager compile of every length first (see _precompile_prefill_lengths); the traced 128-token prefill and the
    # decode trace are then captured by the first pair with all programs already resident.
    compile_times = _precompile_prefill_lengths(
        generator,
        models,
        mesh_device,
        model_args,
        tt_kv_cache,
        page_table,
        tokenizer,
        {i for i, _ in pairs},
        max_seq_len,
    )
    rows, all_failures, warmed_lengths = [], [], set()
    for isl, osl in pairs:
        prompts = _prompts_for(isl, batch_size)
        logger.info(f"== batch {batch_size} ISL {isl} OSL {osl}")
        metrics, outputs = _run_pair(
            generator,
            models,
            mesh_device,
            model_args,
            tt_kv_cache,
            page_table,
            tokenizer,
            prompts,
            isl,
            osl,
            sampling,
            max_seq_len,
            warmed_lengths,
            compile_times,
        )
        failures, info = _check_outputs(tokenizer, vocab_size, isl, outputs, stop_ids, first_ids, content_id)
        row = {**meta, **metrics, **info, "status": "FAIL" if failures else "ok"}
        rows.append(row)
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            f"   prefill {metrics['prefill_total_s']:.2f}s ({metrics['prefill_per_user_ms']:.0f} ms/user), "
            f"decode {metrics['decode_step_mean_ms']:.1f} ms/step (p99 {metrics['decode_step_p99_ms']:.1f}), "
            f"{metrics['tok_s_user']} tok/s/user, {metrics['tok_s_aggregate']} tok/s aggregate; "
            f"{info['content_reached']}/{batch_size} users reached {CONTENT_TOKEN}; "
            f"{'; '.join(failures) if failures else 'checks ok'} | {info.get('sample_output', '')[:100]!r}"
        )
        all_failures.extend(f"ISL {isl} OSL {osl}: {msg}" for msg in failures)

    logger.info(
        f"\nSolar-Open multi-user regression, {model_name} on {meta['mesh']} ({meta['git']}{tag}), batch {batch_size}, "
        f"context {max_seq_len} tokens/user -> {out_file}:\n" + _markdown_table(rows)
    )
    assert not all_failures, "Regression failures:\n" + "\n".join(all_failures)
