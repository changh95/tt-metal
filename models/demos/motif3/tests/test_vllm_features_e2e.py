# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end validation of a RUNNING Motif-3 vLLM server: chunked prefill, prefix caching and MTP self-speculative
decoding (``docs/features/FEATURES_DESIGN.md`` §5.4; results in ``docs/FEATURES_RESULTS.md``), packed multi-row prefill
(P5) and the 64-row speculative verify (T64) (``docs/p5_t64/P5_T64_DESIGN.md`` §6 gates E2E-P, G-serve, E2E-X; results
in ``docs/P5_T64_RESULTS.md``).

This module is an HTTP client only: it never imports ttnn or opens a device. Run it with the devices hidden (the
tt-metal root conftest opens the UMD cluster even for collection, which would disturb the server that owns the chips)::

    MOTIF3_E2E_URL=http://127.0.0.1:8021 MOTIF3_E2E_PROFILE=dsamp_pk MOTIF3_E2E_OUT=<dir> \\
    MOTIF3_E2E_REFERENCE=<dir of a dsamp run> MOTIF3_E2E_SERVER_LOG=<vllm log> \\
    scripts/hostrun.sh -t 9000 -n e2e_dsamp_pk -- python -m pytest -p no:cacheprovider -s -q --timeout=0 \\
        -o addopts="" models/demos/motif3/tests/test_vllm_features_e2e.py

(``logs/serve/p5_t64/run_e2e.sh PROFILE [REFERENCE]`` sets all of this up.)

Environment:

* ``MOTIF3_E2E_URL`` (default ``http://127.0.0.1:8000``): the server. Live tests skip when it does not answer.
* ``MOTIF3_E2E_PROFILE``: what the server under test enables; the assertions follow it.

  - The features track's launches (``logs/serve/features/features_hold.sh``, host sampling): ``all`` = the opt-in MTP
    launch (chunked prefill + prefix caching + MTP K=1, ``packed`` verify; NOT the production launch, which has no
    ``--speculative-config``), ``nomtp`` (no ``--speculative-config``), ``off`` (``MOTIF3_*=0`` with the draft-1 TIS
    flags). Their FEATURES_RESULTS runs served the A = 64 tree at budget 8128: re-run them with
    ``MOTIF3_E2E_BUDGET=8128`` if the launch still passes 8128.
  - The P5 / T64 launches (``logs/serve/p5_t64/p5t64_hold.sh``; chunk budget = threshold = 8064, device sampling):
    ``dsamp`` (the production default: chunked prefill + prefix caching + exact device sampling, no speculation;
    packed prefill OFF = the per-row reference path), ``dsamp_pk`` (+ ``MOTIF3_PACKED_PREFILL=1``), ``mtp_auto``
    (``dsamp_pk`` + the opt-in MTP launch with ``MOTIF3_SPEC_VERIFY=auto``: everything on, E2E-X), ``mtp_packed`` (MTP
    with today's idle-lane verify), ``mtp_wide`` (MTP with the one-trace T64 fallback).
* ``MOTIF3_E2E_BUDGET`` (default 8064 = ``prefill_plan.recommended_budget(8192, A = 128)``): the chunk budget =
  long-prefill threshold the features line must show on a chunked launch.
* ``MOTIF3_E2E_OUT``: directory for one JSON per test (raw results and summaries).
* ``MOTIF3_E2E_REFERENCE``: the ``MOTIF3_E2E_OUT`` of a run on another server. The ``test_9*`` comparison tests read
  both directories; they need no server, except that a non-speculating live server answers the near-tie probes.
* ``MOTIF3_E2E_SERVER_LOG``: the server's log: the bridge's feature line, the warm-up / capture lines, the plugin's PS-1
  canary (a verify step that held a sampled request logs "not speculable"), the shutdown counters (``test_99``).
* ``MOTIF3_E2E_SMOKE_REFERENCE`` (default ``logs/serve/results/final/smoke.json``): the draft-1 TIS server's chat
  outputs (SERVING_SMOKE §3; 8/8 greedy texts character-identical to FULL_MODEL_VALIDATION §3).
* ``MOTIF3_E2E_TPUT_LEVELS`` (default ``1,8,16,20,24,32``), ``MOTIF3_E2E_TPUT_TOKENS`` (256),
  ``MOTIF3_E2E_TPUT_THINKING`` (``on``): the decode-throughput levels.
* ``MOTIF3_E2E_GATE_TOKENS`` (4000), ``MOTIF3_E2E_NEAR_TIE`` (1.25): see "Gated bursts" and "Comparisons" below.

Gated bursts. Packed prefill makes a row's numerics depend on the pass it runs in (the row-local programs run at M = T
rows; P5_T64_DESIGN §3.6, gate CP-P: per-row PCC ~0.997 vs solo, the same floor as the row at another bucket). A burst
that vLLM splits differently over its prefill steps therefore gives other near-tie flips. A *gated* burst first sends
a gate request (a cold ~4K-token prompt, ``max_tokens`` 1) and submits its requests ~20 ms apart while the gate's
prefill runs; the plugin's prefill steps hide running decodes, so every server prefills the burst in the same steps
with the same rows, and two launches with the same prefill path must give the same tokens.

Live tests (in order; each records ``<name>.json``):

1. ``test_server_config``: ``/v1/models``; the bridge's ``Motif-3 features:`` line matches the profile (budget,
   ``spec_verify``, ``packed_prefill``, ``c*``); packed warm-up and T64 capture lines.
2. ``test_smoke_chats``: the SERVING_SMOKE §3 workload (4 prompts x greedy / T=1.0 / T=0.6 seeded x thinking off /
   on; 12 gated each): answers judged correct, greedy texts identical to the draft-1 reference on per-row launches
   (packed launches: recorded; ``test_92`` checks them against the reference launch), sampled texts never the greedy
   text.
3. ``test_greedy_speculation_lossless``: the 8 greedy chats alone (4 gated, thinking off then on), so every row
   speculates on an MTP launch: drafts and acceptance from ``/metrics``.
4. ``test_prefix_cache_hit``: a ~6K-token system prompt + question twice (second TTFT far lower, identical text,
   ``vllm:prefix_cache_hits``), then a different question behind the same system prompt.
5. ``test_multi_turn_decode_written_hit``: turn 2 = turn 1's prompt + its 192 DECODE-generated tokens + a new turn:
   the hit covers decode-written blocks (needs KV-R) and must equal the same prompt sent cold (``cache_salt``).
6. ``test_long_prompt_chunked``: a ~30K-token needle prompt while another request decodes (chunked prefill bounds the
   stall), then the same prompt again (a full prefix hit).
7. ``test_mixed_concurrency_ps1``: 16 sampled (seeded) + 16 greedy needle / arithmetic requests: PS-1 (no draft while
   a sampled request is live), then the 16 greedy alone (drafts resume, prefix hits).
   ``test_spec_refuses_logprobs``: ``logprobs`` refused (HTTP 400) on a speculating launch, served otherwise.
8. ``test_burst_ttft``: 32 short unique prompts at once, then 32 sharing a ~2K-token system prompt (FR §3.7): TTFT;
   E2E-P bars on packed launches (TTFT mean <= 2.8 s / <= 6.0 s, the packed / pk1 step <= 2.0 s / <= 3.5 s).
9. ``test_decode_throughput``: greedy, thinking on, 256 tokens (ignore_eos), gated, c = 1 / 8 / 16 / 20 / 24 / 32:
   per-user TPOT, the steady-window aggregate tokens/s, drafts and acceptance; T64 launches draft at every level.
10. ``test_p5_gated_bursts``: 32 short prompts (and the same batch again: bitwise repeat), 32 sharing a 2K prefix (one
    step: same-step hits, a pk1 pass), 8 identical 6K prompts (same-step hits), the SERVING_SMOKE 32-request mixed
    burst (greedy + seeded), a 16-session second turn at one common start (pk1, distinct tails): answers, step times.
11. ``test_p5_decode_stall``: a request decoding while 31 short prompts arrive, and 16 decoding while 16 arrive: the
    decoding requests' worst inter-token gap (E2E-P bar 2.5 s on packed launches).
12. ``test_p5_long_beside_burst``: a ~30K-token prompt with 16 short prompts arriving while its chunks run.
13. ``test_t64_ps1_mix``: 8 seeded sampled + 24 greedy requests at c = 32 (gated): no draft while a sampled request is
    live; on T64 launches the 24 greedy lanes draft (T64 verify) once the sampled ones finished.
14. ``test_p5_pass_size_floor``: the 32 short prompts in gated groups of B = 4 / 8 / 16 / 32 (a packed launch runs pk0
    passes of T = 256 / 512 / 1024 / 2048 rows): per-row launches must give the same tokens for every B (batch
    invariance); on packed launches the differences between B measure the M = T matmul floor.
15. ``test_p5_logprob_replay`` (non-speculating launches): the gated 32 short prompts again with ``top_logprobs`` 5, so
    ``test_94`` can read both launches' log-probabilities where they diverge; the tokens must equal the device-sampled
    run's (``test_p5_gated_bursts``).

Comparison tests (``MOTIF3_E2E_OUT`` vs ``MOTIF3_E2E_REFERENCE``): ``test_90`` / ``test_91`` (prefix hit, multi-turn,
long prompt, mixed greedy rows), ``test_92`` (every gated workload), ``test_93`` (decode throughput: speedups vs the
reference launch and the tokens of every level), ``test_94`` (the logprob replays). Two launches with the same prefill
path (both per-row, or both packed, and gated) must give identical greedy tokens, and identical seeded sampled tokens
when both sample on the device: that is the "token-exact vs no-spec" check of an MTP launch against its
non-speculating twin. Between a packed and a per-row launch every greedy difference must be a near-tie (the floor):
at the first divergence both tokens are among the top 3 and their log-probabilities lie within
``MOTIF3_E2E_NEAR_TIE`` (1.25) -- in both launches' own logprobs (``test_94``), or in a probe a non-speculating live
server answers (prompt + the common prefix, ``logprobs``; ``test_91`` - ``test_93``). The bar is the floor measured
on this tree: the LM head's logits are bf16 (one ulp is 0.125 at |logit| 16-32; the margins come in multiples of
0.125), a packed row's logits depend on its pass's T (the row-local programs at bucket T; gate CP-P: packed vs per-row
is the same PCC ~0.997 as per-row at bucket T vs per-row), and the same 32 prompts in packed passes of different T
(``test_p5_pass_size_floor``) diverge at first-divergence margins up to 1.125 (9 ulps; 3 of 105, one prompt's first
token), so 1.25 = the floor's maximum + one ulp.

``test_99_shutdown_counters`` (after the server's SIGTERM shutdown): the generator's and the bridge's counters
(``packed_passes``, ``packed_solo_fallbacks`` 0, ``wide_steps``, ``nongreedy_verify_rows`` 0, ...), the PS-1 canary
and a clean device close.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

PROJECT = Path(__file__).resolve().parents[5]  # .../motif-3 (tests -> motif3 -> demos -> models -> tt-metal -> root)
BASE = os.environ.get("MOTIF3_E2E_URL", "http://127.0.0.1:8000").rstrip("/")
PROFILE = os.environ.get("MOTIF3_E2E_PROFILE", "all").strip().lower()
OUT_DIR = os.environ.get("MOTIF3_E2E_OUT")
REF_DIR = os.environ.get("MOTIF3_E2E_REFERENCE")
SERVER_LOG = os.environ.get("MOTIF3_E2E_SERVER_LOG")
SMOKE_REF = Path(
    os.environ.get("MOTIF3_E2E_SMOKE_REFERENCE", str(PROJECT / "logs" / "serve" / "results" / "final" / "smoke.json"))
)
MODEL = "Motif-Technologies/Motif-3"
REVISION = "2ed2ed5cfabffa10fdabb2fc0d0288f8e6de893a"
SYSTEM = "You are a helpful assistant."
EOS_IDS = {0, 3, 6}  # generation_config.json eos_token_id: <|endoftext|>, <|user|>, <|endofturn|>
CHAT, COMP = "/v1/chat/completions", "/v1/completions"
# chunked / prefix: the vLLM features; spec: --speculative-config (MTP K = 1); dsamp: exact device sampling
# ("sample_on_device_mode": "decode_only"); packed: MOTIF3_PACKED_PREFILL=1 (P5); verify: MOTIF3_SPEC_VERIFY (T64)
PROFILES = {
    # logs/serve/features/features_hold.sh (docs/FEATURES_RESULTS.md); "all" is the opt-in MTP launch
    "all": dict(chunked=True, prefix=True, spec=True, dsamp=False, packed=False, verify="packed"),
    "nomtp": dict(chunked=True, prefix=True, spec=False, dsamp=False, packed=False, verify=None),
    "off": dict(chunked=False, prefix=False, spec=False, dsamp=False, packed=False, verify=None),
    # logs/serve/p5_t64/p5t64_hold.sh (docs/P5_T64_RESULTS.md); "dsamp" is today's production default
    "dsamp": dict(chunked=True, prefix=True, spec=False, dsamp=True, packed=False, verify=None),
    "dsamp_pk": dict(chunked=True, prefix=True, spec=False, dsamp=True, packed=True, verify=None),
    "mtp_auto": dict(chunked=True, prefix=True, spec=True, dsamp=True, packed=True, verify="auto"),
    "mtp_packed": dict(chunked=True, prefix=True, spec=True, dsamp=True, packed=True, verify="packed"),
    "mtp_wide": dict(chunked=True, prefix=True, spec=True, dsamp=True, packed=True, verify="wide"),
}
if PROFILE not in PROFILES:
    raise ValueError(f"MOTIF3_E2E_PROFILE must be one of {sorted(PROFILES)}, got {PROFILE!r}")
FEAT = PROFILES[PROFILE]
SPEC = FEAT["spec"]
WIDE = SPEC and FEAT["verify"] in ("auto", "wide")  # the 64-row verify serves the verify steps the idle lanes cannot
BLOCK = 64
# --max-num-batched-tokens = --long-prefill-token-threshold of the chunked launches: prefill_plan.recommended_budget(
# span cap 8192, A) = 8064 with A = 128 (gate G9's per-bucket sp1 q / k, lead decision F5); the launchers derive it
# from the tree they serve. The features track's FEATURES_RESULTS runs (A = 64) used 8128.
BUDGET = int(os.environ.get("MOTIF3_E2E_BUDGET", "8064"))
GATE_TOKENS = int(os.environ.get("MOTIF3_E2E_GATE_TOKENS", "4000"))
# the largest margin of a packed-vs-per-row flip: the packing floor measured between packed passes of different T
# (1.125 = 9 bf16 ulps of a |logit| in 16-32) + one ulp (module docstring, "Comparisons")
NEAR_TIE = float(os.environ.get("MOTIF3_E2E_NEAR_TIE", "1.25"))
NEAR_TIE_RANK = 3  # ... and both tokens among the top 3 there
# E2E-P / G-serve bars (P5_T64_DESIGN.md §1, §6.2)
BAR_BURST_A_TTFT_MEAN_S = 2.8  # (a) 32 unique short prompts at once: client TTFT mean (today 20.56 s)
BAR_BURST_B_TTFT_MEAN_S = 6.0  # (b) 32 prompts sharing a 2K system prompt (today 23.46 s)
BAR_PACKED_STEP_A_S = 2.0  # the packed prefill step of (a) (today ~20.6 s)
BAR_PK1_STEP_B_S = 3.5  # the pk1 step of (b)
BAR_STALL_S = 2.5  # a decoding request's worst inter-token gap while a short burst prefills (today ~0.65 s x rows)
BAR_C32_SPEEDUP = 1.4  # c = 32 greedy tok/s of a T64 launch vs the same launch without MTP (expected ~1.7x)
BAR_LOW_C_SPEEDUP = 1.5  # c = 1 / 8 sanity floor (expected ~1.9x, FR §3.8)

# The SERVING_SMOKE §3 prompts and sampling modes (logs/serve/clients/smoke_tests.py), verbatim.
SMOKE_PROMPTS = {
    "en_explain": "Explain why the sky is blue during the day but often red or orange at sunset. Keep it to one short "
    "paragraph.",
    "ko_question": "대한민국의 수도는 어디이며, 그 도시가 역사적으로 중요한 이유를 두세 문장으로 설명해 주세요.",
    "gsm8k_math": "A bakery sells muffins for $3 each and cookies for $2 each. On Monday it sold 45 muffins and twice "
    "as many cookies as muffins. On Tuesday it sold 30 muffins and 50 cookies. How much money did the bakery make in "
    "total over the two days? Show your work and give the final answer.",
    "python_task": "Write a Python function `is_palindrome(s: str) -> bool` that returns True if the string is a "
    "palindrome, ignoring case and non-alphanumeric characters. Include a docstring and two example calls.",
}
SMOKE_MODES = {
    "greedy": dict(temperature=0.0, top_p=1.0),
    "t1.0_p0.95": dict(temperature=1.0, top_p=0.95, seed=1234),
    "t0.6_p0.95": dict(temperature=0.6, top_p=0.95, seed=1234),
}
QUESTIONS = [
    "What are the main differences between TCP and UDP? Answer in a few bullet points.",
    "Write a haiku about autumn leaves, then explain the imagery you used.",
    "한국의 전통 음식 세 가지를 소개하고 각각의 특징을 설명해 주세요.",
    "Solve for x: 3x + 7 = 2x - 5. Explain each step.",
    "Write a Python function that returns the n-th Fibonacci number using memoization, with a short docstring.",
    "Summarize the causes of the French Revolution in one paragraph.",
    "What is the difference between a list and a tuple in Python? Give an example of each.",
    "Explain how photosynthesis works to a ten-year-old.",
    "A train travels 180 km in 2.5 hours. What is its average speed in km/h and in m/s?",
    "Translate into English and explain the meaning of the Korean proverb '천 리 길도 한 걸음부터'.",
    "Write a SQL query that returns the top 5 customers by total order amount from tables customers(id, name) and "
    "orders(id, customer_id, amount).",
    "Explain the difference between supervised and unsupervised learning with one example each.",
    "Describe three ways to reduce the memory use of a Python program.",
    "Why do we have seasons on Earth? Explain briefly.",
    "Write a short motivational message for a student preparing for exams.",
    "List five common HTTP status codes and what they mean.",
    "What is a binary search tree? Describe insertion in a few sentences.",
    "서울과 부산의 차이점을 세 가지로 설명해 주세요.",
    "Explain what a hash table is and why lookups are fast on average.",
    "Give a recipe outline for a simple tomato pasta.",
    "What is the Pythagorean theorem? Give a worked example.",
    "Explain the concept of opportunity cost with an everyday example.",
    "Write a limerick about a cat who loves to code.",
    "What are the benefits and drawbacks of remote work? Answer in bullet points.",
    "Explain recursion to a beginner programmer, with a tiny example.",
    "Describe the water cycle in four steps.",
    "What is the difference between weather and climate?",
    "Write a polite email declining a meeting invitation.",
    "Explain why the moon has phases.",
    "What does the `git rebase` command do, and when should you avoid it?",
    "인공지능이 일상생활에 미치는 영향을 짧게 설명해 주세요.",
    "Compare electric cars and gasoline cars in terms of cost, maintenance and environmental impact.",
]
CITIES = [
    "Lisbon",
    "Osaka",
    "Denver",
    "Nairobi",
    "Hanoi",
    "Tallinn",
    "Quito",
    "Perth",
    "Busan",
    "Lyon",
    "Cork",
    "Accra",
]
GOODS = ["copper wire", "olive oil", "ceramic tiles", "wool blankets", "solar panels", "coffee beans", "steel bolts",
         "paper rolls", "rubber seals", "glass jars", "cotton thread", "brass valves"]  # fmt: skip
WORDS = ["amber", "falcon", "cobalt", "willow", "ember", "glacier", "harbor", "juniper", "lantern", "meadow", "onyx",
         "quartz"]  # fmt: skip


def log(msg: str) -> None:
    print(f"[e2e {PROFILE} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ======================================================================================================================
# tokenizer and prompts
# ======================================================================================================================
_TOK = None


def tokenizer():
    """The server's tokenizer (the HF snapshot at the pinned revision, offline)."""
    global _TOK
    if _TOK is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", str(PROJECT / "hf_home"))
        from transformers import AutoTokenizer

        snap = PROJECT / "hf_home" / "hub" / "models--Motif-Technologies--Motif-3" / "snapshots" / REVISION
        src = str(snap) if (snap / "tokenizer.json").is_file() else MODEL
        _TOK = AutoTokenizer.from_pretrained(src, trust_remote_code=True, revision=None if snap.is_dir() else REVISION)
    return _TOK


def chat_ids(messages: Sequence[Dict[str, str]], thinking: bool = False) -> List[int]:
    """The chat-templated prompt token ids (what the server builds for /v1/chat/completions)."""
    ids = tokenizer().apply_chat_template(
        list(messages), add_generation_prompt=True, enable_thinking=thinking, tokenize=True
    )
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return [int(t) for t in ids]


def _record(rng: random.Random, i: int) -> str:
    return (
        f"Record {i:05d}: the warehouse in {rng.choice(CITIES)} received {rng.randint(10, 990)} crates of "
        f"{rng.choice(GOODS)} on day {rng.randint(1, 365)}, inspected by clerk #{rng.randint(100, 999)}."
    )


def records_text(target_tokens: int, seed: int) -> str:
    """Synthetic log of about ``target_tokens`` tokens (deterministic in ``seed``)."""
    rng = random.Random(seed)
    per = len(tokenizer().encode(_record(random.Random(1), 1) + "\n", add_special_tokens=False))
    n = max(1, int(target_tokens) // per)
    return "\n".join(_record(rng, i) for i in range(n))


def needle_prompt(target_tokens: int, seed: int, depth: float = 0.5):
    """User message of ~``target_tokens`` tokens hiding a passphrase; returns ``(text, answer)``
    (``logs/serve/clients/haystack.py``)."""
    rng = random.Random(seed)
    code = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
    needle = f"IMPORTANT NOTE: the secret passphrase for vault {seed} is '{code}'."
    question = (
        f"\n\nQuestion: according to the records above, what is the secret passphrase for vault {seed}? Reply with "
        f"the passphrase only."
    )
    tok = tokenizer()
    per = len(tok.encode(_record(random.Random(1), 1) + "\n", add_special_tokens=False))
    overhead = len(tok.encode(needle + question, add_special_tokens=False)) + 40
    n = max(1, (int(target_tokens) - overhead) // per)
    recs = [_record(rng, i) for i in range(n)]
    recs.insert(int(depth * len(recs)), needle)
    return "Below is a log of warehouse records.\n\n" + "\n".join(recs) + question, code


def bos_id() -> int:
    """``<|beginoftext|>`` (the chat template's first token) for raw token-id completions."""
    return int(tokenizer().convert_tokens_to_ids("<|beginoftext|>"))


# ======================================================================================================================
# HTTP client (aiohttp, SSE streaming with per-chunk arrival times)
# ======================================================================================================================
def _timeout():
    import aiohttp

    return aiohttp.ClientTimeout(total=7200, sock_read=7200)


def chat_body(user: str, *, temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 512,
              thinking: Optional[bool] = False, system: Optional[str] = SYSTEM, **extra) -> Dict[str, Any]:  # fmt: skip
    """A /v1/chat/completions body (thinking ``None`` = the server default, on)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    b: Dict[str, Any] = dict(model=MODEL, messages=msgs, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    if thinking is not None:
        b["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    b.setdefault("return_token_ids", True)
    b.setdefault("skip_special_tokens", False)
    b.update(extra)
    return b


def comp_body(prompt, **kw) -> Dict[str, Any]:
    """A /v1/completions body (greedy unless ``kw`` says otherwise)."""
    b: Dict[str, Any] = dict(model=MODEL, prompt=prompt, temperature=0.0, return_token_ids=True,
                             skip_special_tokens=False)  # fmt: skip
    b.update(kw)
    return b


def _token_id(tok: str) -> Optional[int]:
    """``token_id:<n>`` (``return_tokens_as_token_ids``) -> n."""
    return int(tok.split(":", 1)[1]) if isinstance(tok, str) and tok.startswith("token_id:") else None


async def stream(session, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Streaming POST; returns status, text, token_ids, usage, finish_reason, chunk_times (s after send), ttft_s,
    latency_s, t_send / t_first / t_last (perf_counter), error; with chat ``logprobs`` / ``top_logprobs`` and
    ``return_tokens_as_token_ids`` also ``top_logprobs``: per generated token, ``[[token id, logprob], ...]``."""
    body = dict(body, stream=True)
    body.setdefault("stream_options", {"include_usage": True})
    res: Dict[str, Any] = dict(status=None, text="", token_ids=[], usage=None, finish_reason=None, chunk_times=[],
                               chunk_ntok=[], error=None)  # fmt: skip
    if body.get("top_logprobs"):
        res["top_logprobs"] = []
    t0 = time.perf_counter()
    res["t_send"] = t0
    try:
        async with session.post(BASE + path, json=body) as r:
            res["status"] = r.status
            if r.status != 200:
                res["error"] = (await r.text())[:3000]
                return res
            buf = b""
            async for chunk in r.content.iter_any():
                buf += chunk
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    for line in event.split(b"\n"):
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            continue
                        now = time.perf_counter() - t0
                        js = json.loads(data)
                        if js.get("error"):
                            res["error"] = json.dumps(js["error"])[:3000]
                        if js.get("usage"):
                            res["usage"] = js["usage"]
                        for ch in js.get("choices") or []:
                            if "delta" in ch:
                                d = ch["delta"] or {}
                                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                            else:
                                piece = ch.get("text") or ""
                            tids = ch.get("token_ids") or []
                            if piece or tids:
                                res["chunk_times"].append(now)
                                res["chunk_ntok"].append(len(tids))
                                res["text"] += piece
                                res["token_ids"].extend(int(t) for t in tids)
                            if "top_logprobs" in res and "delta" in ch:
                                for e in (ch.get("logprobs") or {}).get("content") or []:
                                    res["top_logprobs"].append(
                                        [
                                            [_token_id(t["token"]), float(t["logprob"])]
                                            for t in e.get("top_logprobs") or []
                                        ]
                                    )
                            if ch.get("finish_reason") is not None:
                                res["finish_reason"] = ch["finish_reason"]
    except Exception as exc:  # noqa: BLE001 - reported in the result
        res["error"] = f"{type(exc).__name__}: {exc}"
    ct = res["chunk_times"]
    res["latency_s"] = time.perf_counter() - t0
    res["ttft_s"] = ct[0] if ct else None
    res["t_first"] = t0 + ct[0] if ct else None
    res["t_last"] = t0 + ct[-1] if ct else None
    return res


async def get_json(session, path: str):
    async with session.get(BASE + path) as r:
        r.raise_for_status()
        return await r.json()


async def metrics(session) -> Dict[str, float]:
    """Counter / gauge values of ``/metrics`` (``vllm:*``; labels summed)."""
    async with session.get(BASE + "/metrics") as r:
        txt = await r.text()
    out: Dict[str, float] = {}
    for line in txt.splitlines():
        if not line.startswith("vllm:"):
            continue
        name_labels, _, val = line.rpartition(" ")
        name = name_labels.split("{", 1)[0]
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


METRIC_KEYS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
)


def mdelta(m1: Dict[str, float], m0: Dict[str, float]) -> Dict[str, float]:
    d = {k.replace("vllm:", "").replace("_total", ""): m1.get(k, 0.0) - m0.get(k, 0.0) for k in METRIC_KEYS}
    if d["spec_decode_num_draft_tokens"] > 0:
        d["acceptance"] = d["spec_decode_num_accepted_tokens"] / d["spec_decode_num_draft_tokens"]
    n = d.pop("request_prefill_time_seconds_count")
    s = d.pop("request_prefill_time_seconds_sum")
    d["prefill_requests"] = n
    d["prefill_s_mean"] = s / n if n else None  # the server's per-request prefill time (scheduled -> first token)
    return d


def tpot(r: Dict[str, Any]) -> Optional[float]:
    """(t_last - t_first) / (completion_tokens - 1) (vLLM bench's TPOT)."""
    n = (r.get("usage") or {}).get("completion_tokens")
    ct = r.get("chunk_times") or []
    if not n or n < 2 or len(ct) < 2:
        return None
    return (ct[-1] - ct[0]) / (n - 1)


def max_gap(r: Dict[str, Any], lo: Optional[float] = None, hi: Optional[float] = None) -> Optional[float]:
    """Worst inter-chunk gap of a streamed request (absolute perf_counter window ``[lo, hi]`` if given)."""
    ts = [r["t_send"] + c for c in r.get("chunk_times") or []]
    gaps = [b - a for a, b in zip(ts, ts[1:]) if (lo is None or b >= lo) and (hi is None or a <= hi)]
    return max(gaps) if gaps else None


def stats(vals: Sequence[Optional[float]]) -> Dict[str, Any]:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {"n": 0}

    def q(p):
        k = (len(v) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)

    return dict(n=len(v), mean=sum(v) / len(v), p50=q(0.5), p90=q(0.9), p99=q(0.99), min=v[0], max=v[-1])


def answer_of(r: Dict[str, Any]) -> str:
    """The text up to the first stop token (and after ``</think>``), special tokens stripped."""
    text = r.get("text") or ""
    text = text.split("</think>", 1)[1] if "</think>" in text else text
    for stop in ("<|endofturn|>", "<|endoftext|>", "<|user|>"):
        text = text.split(stop, 1)[0]
    return text.strip()


def ids_until_eos(ids: Sequence[int]) -> List[int]:
    out = []
    for t in ids:
        out.append(int(t))
        if int(t) in EOS_IDS:
            break
    return out


def first_divergence(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def judge(name: str, text: str) -> Dict[str, Any]:
    """The SERVING_SMOKE §3 correctness checks."""
    a = text
    if name == "en_explain":
        return {"ok": "scatter" in a.lower() and ("sunset" in a.lower() or "longer" in a.lower())}
    if name == "ko_question":
        letters = [c for c in a if c.isalpha()]
        ratio = sum(1 for c in letters if "가" <= c <= "힣") / max(1, len(letters))
        return {"ok": "서울" in a and ratio > 0.8, "hangul_ratio": round(ratio, 3)}
    if name == "gsm8k_math":
        return {"ok": "505" in a}
    if name == "python_task":
        m = re.findall(r"```(?:python)?\n(.*?)```", a, flags=re.S)
        if not m:
            return {"ok": False, "reason": "no code block"}
        extra = (
            "\nassert is_palindrome('A man, a plan, a canal: Panama') is True\nassert is_palindrome('race a car') is "
            "False\nassert is_palindrome('') is True\nassert is_palindrome('No lemon, no melon') is True\nassert "
            "is_palindrome('ab') is False\nprint('EXTRA_ASSERTS_OK')\n"
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "answer.py"
            path.write_text(m[-1] + "\n" + extra)
            p = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=30, cwd=d)
        return {"ok": p.returncode == 0 and "EXTRA_ASSERTS_OK" in p.stdout, "stderr": p.stderr[-300:]}
    return {"ok": None}


def slim(r: Dict[str, Any]) -> Dict[str, Any]:
    """A result without the per-chunk arrays (kept: counts, TTFT, TPOT, the worst gap)."""
    out = {k: v for k, v in r.items() if k not in ("chunk_times", "chunk_ntok", "t_send", "t_first", "t_last")}
    ntok = r.get("chunk_ntok") or []
    out["multi_token_chunks"] = sum(1 for k in ntok if k > 1)
    out["tpot_s"] = tpot(r)
    out["max_gap_s"] = max_gap(r)
    out["completion_tokens"] = (r.get("usage") or {}).get("completion_tokens")
    out["prompt_tokens"] = (r.get("usage") or {}).get("prompt_tokens")
    return out


def save(name: str, payload: Dict[str, Any]) -> None:
    payload = dict(payload, profile=PROFILE, url=BASE, saved=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if OUT_DIR:
        d = Path(OUT_DIR)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    log(f"{name}: {json.dumps(payload.get('summary', {}), ensure_ascii=False)[:2000]}")


def load(dirname: Optional[str], name: str) -> Optional[Dict[str, Any]]:
    if not dirname:
        return None
    p = Path(dirname) / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def server_log_text() -> Optional[str]:
    if not SERVER_LOG or not Path(SERVER_LOG).is_file():
        return None
    return Path(SERVER_LOG).read_text(errors="replace")


# ======================================================================================================================
# fixtures
# ======================================================================================================================
def _server_up() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def server():
    if not _server_up():
        pytest.skip(f"no Motif-3 vLLM server answers at {BASE}/health")
    return BASE


def run(coro):
    return asyncio.run(coro)


async def _session_do(fn):
    import aiohttp

    async with aiohttp.ClientSession(timeout=_timeout()) as s:
        return await fn(s)


# ======================================================================================================================
# gated bursts (deterministic prefill steps on every launch), the idle server, near-tie probes
# ======================================================================================================================
_GATE: List[int] = []


def gate_prompt() -> List[int]:
    """The gate request's prompt: ``GATE_TOKENS`` token ids of a synthetic log (cold every time: its own salt)."""
    if not _GATE:
        ids = tokenizer().encode(records_text(GATE_TOKENS + 400, seed=4242), add_special_tokens=False)
        _GATE.extend([bos_id()] + ids[: GATE_TOKENS - 1])
    return list(_GATE)


async def wait_idle(s, timeout: float = 900.0) -> bool:
    """Until the server runs and queues no request (``vllm:num_requests_running`` / ``_waiting`` both 0)."""
    t_end = time.perf_counter() + timeout
    while True:
        m = await metrics(s)
        if m.get("vllm:num_requests_running", 0.0) == 0 and m.get("vllm:num_requests_waiting", 0.0) == 0:
            return True
        if time.perf_counter() > t_end:
            return False
        await asyncio.sleep(0.5)


async def _poll_drafts(s, stop: asyncio.Event, out: List[Tuple[float, float]], period: float) -> None:
    while not stop.is_set():
        m = await metrics(s)
        out.append((time.perf_counter(), m.get("vllm:spec_decode_num_drafts_total", 0.0)))
        try:
            await asyncio.wait_for(stop.wait(), timeout=period)
        except asyncio.TimeoutError:
            pass


async def gated(s, jobs: Sequence[Tuple[str, Dict[str, Any]]], *, stagger: float = 0.02, settle: float = 0.5,
                poll: Optional[float] = None) -> Dict[str, Any]:  # fmt: skip
    """Submit ``jobs`` (``(path, body)``) behind a gate: the gate request (``GATE_TOKENS`` cold tokens, ``max_tokens``
    1) occupies the engine's prefill for ~3 s while the jobs are submitted ``stagger`` s apart, so they are all queued,
    in that order, before the scheduler's next decision. The plugin's prefill steps hide running decodes, so the jobs
    run in the same prefill steps (same rows, same packed passes) on every launch. ``gate.ok`` says the gate held:
    every job was submitted before the gate's token and got its first token after it. ``poll``: also sample the
    drafts counter every ``poll`` s after the gate (``polls``: ``(perf_counter, drafts)``)."""
    idle = await wait_idle(s)
    gate_body = comp_body(gate_prompt(), max_tokens=1, cache_salt=f"gate-{uuid.uuid4().hex}")
    t0 = time.perf_counter()
    gate_task = asyncio.ensure_future(stream(s, COMP, gate_body))
    await asyncio.sleep(settle)
    tasks = []
    for path, body in jobs:
        tasks.append(asyncio.ensure_future(stream(s, path, body)))
        await asyncio.sleep(stagger)
    t_sub = time.perf_counter()
    gate = await gate_task
    await asyncio.sleep(0.25)  # the gate's request metrics land before the burst's (a prefill step takes >= 0.6 s)
    m_gate = await metrics(s)
    polls: List[Tuple[float, float]] = []
    stop = asyncio.Event()
    poller = asyncio.ensure_future(_poll_drafts(s, stop, polls, poll)) if poll else None
    results = list(await asyncio.gather(*tasks))
    stop.set()
    if poller is not None:
        await poller
    m_end = await metrics(s)
    gf = gate.get("t_first")
    late = [i for i, r in enumerate(results) if gf is None or (r.get("t_first") is not None and r["t_first"] <= gf)]
    ok = bool(idle and gate["status"] == 200 and not gate["error"] and gf is not None and t_sub < gf - 0.15)
    gate_info = dict(ok=ok and not late, idle=idle, status=gate["status"], error=gate["error"],
                     ttft_s=gate.get("ttft_s"), submit_span_s=round(t_sub - t0, 3),
                     first_token_before_gate=late)  # fmt: skip
    if not gate_info["ok"]:
        log(f"warning: the gate did not hold: {gate_info}")
    return dict(results=results, gate=gate_info, m_gate=m_gate, m_end=m_end, t_gate=gf, polls=polls)


def step_stats(g: Dict[str, Any]) -> Dict[str, Any]:
    """A gated burst's prefill: the server's per-request prefill time (scheduled -> first token; for a one-step burst
    that is the step) and each request's first token after the gate's (the gate's step ends -> the burst's starts)."""
    md = mdelta(g["m_end"], g["m_gate"])
    after = [r["t_first"] - g["t_gate"] for r in g["results"] if r.get("t_first") and g.get("t_gate")]
    return dict(server_prefill_s_mean=md["prefill_s_mean"], server_prefill_requests=md["prefill_requests"],
                n=len(g["results"]), first_token_after_gate_s=stats(after))  # fmt: skip


def gated_record(g: Dict[str, Any], jobs: Sequence[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """What a gated workload saves: the requests (for the cross-launch comparison and its probes) and the results."""
    return dict(paths=[p for p, _ in jobs], requests=[b for _, b in jobs], results=[slim(r) for r in g["results"]],
                gate=g["gate"], step=step_stats(g), metrics=mdelta(g["m_end"], g["m_gate"]))  # fmt: skip


async def prompt_ids_of(s, body: Dict[str, Any]) -> Optional[List[int]]:
    """The prompt token ids the server builds for ``body`` (vLLM ``/tokenize``; a token-id prompt as is)."""
    if "messages" not in body:
        p = body["prompt"]
        if isinstance(p, list):
            return [int(t) for t in p]
        req: Dict[str, Any] = dict(model=MODEL, prompt=p)
    else:
        req = dict(model=MODEL, messages=body["messages"], add_generation_prompt=True,
                   chat_template_kwargs=body.get("chat_template_kwargs") or {"enable_thinking": True})  # fmt: skip
    async with s.post(BASE + "/tokenize", json=req) as r:
        if r.status != 200:
            return None
        js = await r.json()
    return [int(t) for t in js.get("tokens") or []]


async def near_tie_probe(s, body: Dict[str, Any], ids_a: Sequence[int], ids_b: Sequence[int]) -> Dict[str, Any]:
    """Where two greedy continuations of the same request first differ: the log-probabilities of the two tokens at that
    position, from a solo prefill of prompt + the common prefix on this server (``logprobs`` 20: a non-speculating
    launch). ``margin`` = |lp_a - lp_b| (a token outside the top 20 counts as the 20th log-prob: a lower bound)."""
    d = first_divergence(ids_a, ids_b)
    if d is None:
        return dict(divergence=None)
    if d >= min(len(ids_a), len(ids_b)):
        return dict(divergence=d, margin=None, reason="one is a prefix of the other")
    prompt = await prompt_ids_of(s, body)
    if not prompt:
        return dict(divergence=d, margin=None, reason="no prompt ids (/tokenize)")
    pb = comp_body(prompt + [int(t) for t in ids_a[:d]], max_tokens=1, logprobs=20, return_tokens_as_token_ids=True,
                   cache_salt=f"probe-{uuid.uuid4().hex}")  # fmt: skip
    pb.pop("return_token_ids", None)
    async with s.post(BASE + COMP, json=pb) as r:
        status, txt = r.status, await r.text()
    if status != 200:
        return dict(divergence=d, margin=None, reason=f"HTTP {status}: {txt[:300]}")
    js = json.loads(txt)
    top = js["choices"][0]["logprobs"]["top_logprobs"][0]
    lp = {int(k.split(":", 1)[1]): float(v) for k, v in top.items()}
    a, b = int(ids_a[d]), int(ids_b[d])
    floor = min(lp.values())
    la, lb = lp.get(a, floor), lp.get(b, floor)
    best = max(lp, key=lp.get)
    return dict(divergence=d, token_a=a, token_b=b, lp_a=round(la, 4), lp_b=round(lb, 4), margin=round(abs(la - lb), 4),
                in_top20=[a in lp, b in lp], ranks=[_rank(lp, a), _rank(lp, b)], probe_argmax=best,
                prompt_tokens=len(prompt))  # fmt: skip


def _rank(lp: Dict[int, float], tok: int) -> Optional[int]:
    """1-based rank of ``tok`` among the returned top log-probs (ties share the better rank), None if absent."""
    if tok not in lp:
        return None
    return 1 + sum(1 for v in lp.values() if v > lp[tok])


def near_tie_ok(m: Optional[Dict[str, Any]]) -> bool:
    """A divergence the floor explains: both tokens among the top ``NEAR_TIE_RANK`` and within ``NEAR_TIE``."""
    if not m or m.get("margin") is None:
        return False
    ranks = m.get("ranks") or [None, None]
    return m["margin"] <= NEAR_TIE and all(r is not None and r <= NEAR_TIE_RANK for r in ranks)


# ======================================================================================================================
# 1. server configuration
# ======================================================================================================================
def test_server_config(server):
    async def go(s):
        return await get_json(s, "/v1/models"), await metrics(s)

    models, m = run(_session_do(go))
    ids = [x["id"] for x in models["data"]]
    assert MODEL in ids, ids
    assert [x.get("max_model_len") for x in models["data"]] == [32768]
    summary: Dict[str, Any] = dict(models=ids, max_model_len=32768,
                                   metrics_present=sorted(k for k in m if "spec_decode" in k)[:6])  # fmt: skip
    text = server_log_text()
    if text is not None:
        feat = re.findall(r"Motif-3 features: (.*)", text)
        assert feat, "the bridge's 'Motif-3 features:' line is missing from the server log"
        line = feat[-1]
        summary["features_line"] = line
        want = {
            "chunked_prefill": FEAT["chunked"],
            "prefix_caching": FEAT["prefix"],
            "kv_replicated": FEAT["prefix"],
        }
        for key, val in want.items():
            assert re.search(rf"\b{key}={val}\b", line), (key, val, line)
        assert re.search(rf"\bspec_tokens={1 if SPEC else 0}\b", line), line
        mode = ("all" if FEAT["prefix"] else "row") + ("_split" if SPEC else "")
        assert re.search(rf"\bkv_write={mode}\b", line), (mode, line)
        if FEAT["chunked"]:
            assert f"(budget {BUDGET}, threshold {BUDGET})" in line, (BUDGET, line)
        summary["serving_config_warnings"] = re.findall(r"WARNING.*Motif-3 serving config.*", text)[:5]
        assert not summary["serving_config_warnings"], summary["serving_config_warnings"]
        if "spec_verify=" in line:  # the P5 / T64 bridge
            assert re.search(rf"\bspec_verify={FEAT['verify'] or 'packed'}\b", line), line
            assert re.search(rf"\bpacked_prefill={FEAT['packed']}\b", line), line
            cstar = re.search(r"\bc\*=(\S+)", line)
            summary["c_star"] = cstar.group(1) if cstar else None
            if SPEC and FEAT["verify"] == "auto":
                assert summary["c_star"] not in (None, "n/a", "never") and 17 <= int(summary["c_star"]) <= 32, line
        warm = re.findall(r"warmup prefill (sp[01]) bucket (\d+)", text)
        summary["warmup_prefill_shapes"] = len(warm)
        pk = re.findall(r"warmup packed prefill: (\d+) shapes \('(\w+)'\) in ([0-9.]+) s", text)
        summary["warmup_packed"] = pk[-1] if pk else None
        caps = re.findall(r"decode trace captured: (T32-spec|T64|plain)", text)
        summary["decode_traces"] = caps
        took = re.findall(r"init engine \(profile, create kv cache, warmup model\) took ([0-9.]+)", text)
        summary["init_engine_s"] = float(took[-1]) if took else None
        pool = re.findall(r"Motif-3 KV pool: (.*)", text)
        summary["kv_pool"] = pool[-1] if pool else None
        if SPEC:
            assert summary["kv_pool"] and "MTP" in summary["kv_pool"], summary["kv_pool"]
        if "spec_verify=" in line:
            if FEAT["packed"]:
                assert "create: packed prefill on (P5)" in text and pk and int(pk[-1][0]) > 0, summary
            else:
                assert "create: packed prefill off" in text and not pk, summary
            if WIDE:
                assert "T64" in caps, caps
            if SPEC and FEAT["verify"] != "wide":
                assert "T32-spec" in caps, caps
    save("server_config", dict(summary=summary))


# ======================================================================================================================
# 2. the SERVING_SMOKE §3 chat workload, compared with the draft-1 server
# ======================================================================================================================
def _smoke_reference() -> Optional[Dict[tuple, Dict[str, Any]]]:
    if not SMOKE_REF.is_file():
        return None
    d = json.loads(SMOKE_REF.read_text())
    return {(c["prompt"], c["mode"], c["thinking"]): c for c in d["chat"]}


def _smoke_jobs(modes: Sequence[str], thinking: Optional[bool], max_tokens: int):
    keys = [(p, m, thinking, max_tokens) for m in modes for p in SMOKE_PROMPTS]
    jobs = [(CHAT, chat_body(SMOKE_PROMPTS[p], max_tokens=mt, thinking=th, **SMOKE_MODES[m])) for p, m, th, mt in keys]
    return keys, jobs


def _smoke_record(key, r: Dict[str, Any]) -> Dict[str, Any]:
    pname, mode, thinking, _ = key
    out = slim(r)
    out.update(prompt=pname, mode=mode, thinking=thinking, judge=judge(pname, answer_of(r)))
    out["stopped_on_eos"] = r["finish_reason"] == "stop" and bool(r["token_ids"]) and r["token_ids"][-1] in EOS_IDS
    return out


def _compare_smoke(chats: List[Dict[str, Any]], ref) -> Dict[str, Any]:
    same, diff = [], []
    for c in chats:
        key = (c["prompt"], c["mode"], c["thinking"])
        if ref is None or key not in ref:
            continue
        (same if c["text"] == ref[key]["text"] else diff).append(key)
    return dict(identical=[list(k) for k in same], different=[list(k) for k in diff])


def _draft1_identity_required() -> bool:
    """Greedy texts equal the draft-1 server's (solo prefills) wherever this launch prefills a row alone too: packed
    launches run concurrent rows in packed passes (a row may differ at a near-tie; ``test_92`` checks them against the
    per-row launch instead)."""
    return not FEAT["packed"]


def test_smoke_chats(server):
    async def go(s):
        m0 = await metrics(s)
        ka, ja = _smoke_jobs(SMOKE_MODES, False, 512)
        ga = await gated(s, ja)
        kb, jb = _smoke_jobs(SMOKE_MODES, None, 3072)
        gb = await gated(s, jb)
        return (ka, ja, ga), (kb, jb, gb), mdelta(await metrics(s), m0)

    (ka, ja, ga), (kb, jb, gb), md = run(_session_do(go))
    chats = [_smoke_record(k, r) for k, r in zip(ka + kb, ga["results"] + gb["results"])]
    ref = _smoke_reference()
    cmp = _compare_smoke(chats, ref)
    greedy = [c for c in chats if c["mode"] == "greedy"]
    sampled = [c for c in chats if c["mode"] != "greedy"]
    by_g = {(c["prompt"], c["thinking"]): c["text"] for c in greedy}
    collapsed = [
        (c["prompt"], c["mode"], c["thinking"]) for c in sampled if c["text"] == by_g[(c["prompt"], c["thinking"])]
    ]
    g_cmp = [k for k in cmp["identical"] if k[1] == "greedy"]
    s_cmp = [k for k in cmp["identical"] if k[1] != "greedy"]
    summary = dict(
        gates_ok=[ga["gate"]["ok"], gb["gate"]["ok"]],
        status_ok=sum(c["status"] == 200 and not c["error"] for c in chats),
        judged_ok=sum(bool(c["judge"].get("ok")) for c in chats),
        stopped_on_eos=sum(c["stopped_on_eos"] for c in chats),
        greedy_identical_to_draft1=f"{len(g_cmp)}/{len(greedy)}" if ref else None,
        sampled_identical_to_draft1=f"{len(s_cmp)}/{len(sampled)}" if ref else None,
        different=cmp["different"],
        sampled_equal_to_greedy=collapsed,
        tokens=[c["completion_tokens"] for c in chats],
        metrics=md,
    )
    save("smoke_chats", dict(summary=summary, chats=chats, runs=dict(a=gated_record(ga, ja), b=gated_record(gb, jb))))
    assert summary["status_ok"] == 24, [c["error"] for c in chats if c["error"]]
    assert all(c["judge"].get("ok") for c in greedy), [
        (c["prompt"], c["thinking"]) for c in greedy if not c["judge"]["ok"]
    ]
    assert summary["stopped_on_eos"] == 24
    assert len(collapsed) <= 2, f"sampled outputs equal to the greedy text (argmax collapse?): {collapsed}"
    if ref is not None and _draft1_identity_required():
        assert len(g_cmp) == len(greedy), f"greedy texts differ from the draft-1 server: {cmp['different']}"


# ======================================================================================================================
# 3. greedy-only chats: every row speculates on an MTP launch
# ======================================================================================================================
def test_greedy_speculation_lossless(server):
    async def go(s):
        ka, ja = _smoke_jobs(["greedy"], False, 512)
        ga = await gated(s, ja)
        kb, jb = _smoke_jobs(["greedy"], None, 3072)
        gb = await gated(s, jb)
        return (ka, ja, ga), (kb, jb, gb)

    (ka, ja, ga), (kb, jb, gb) = run(_session_do(go))
    chats = [_smoke_record(k, r) for k, r in zip(ka + kb, ga["results"] + gb["results"])]
    md_off, md_on = mdelta(ga["m_end"], ga["m_gate"]), mdelta(gb["m_end"], gb["m_gate"])
    ref = _smoke_reference()
    cmp = _compare_smoke(chats, ref)
    summary = dict(
        gates_ok=[ga["gate"]["ok"], gb["gate"]["ok"]],
        identical_to_draft1=f"{len(cmp['identical'])}/{len(chats)}" if ref else None,
        different=cmp["different"],
        judged_ok=sum(bool(c["judge"].get("ok")) for c in chats),
        tpot_ms=[round(1e3 * c["tpot_s"], 1) if c["tpot_s"] else None for c in chats],
        tokens=[c["completion_tokens"] for c in chats],
        metrics_thinking_off=md_off,
        metrics_thinking_on=md_on,
    )
    save("greedy_speculation", dict(summary=summary, chats=chats, runs=dict(a=gated_record(ga, ja),
                                                                           b=gated_record(gb, jb))))  # fmt: skip
    assert all(c["status"] == 200 and not c["error"] for c in chats)
    assert all(c["judge"].get("ok") for c in chats)
    if SPEC:
        assert md_off["spec_decode_num_drafts"] > 0 and md_on["spec_decode_num_drafts"] > 0, (md_off, md_on)
        assert md_on.get("acceptance", 0) > 0.5, md_on
    else:
        assert md_off["spec_decode_num_drafts"] == 0 and md_on["spec_decode_num_drafts"] == 0
    if ref is not None and _draft1_identity_required():
        assert not cmp["different"], f"greedy texts differ from the draft-1 server: {cmp['different']}"


# ======================================================================================================================
# 4. prefix caching: the same long system prompt twice
# ======================================================================================================================
def _handbook(target_tokens: int, seed: int) -> str:
    return (
        "You are the operations assistant of a logistics company. Answer from the shipment log below; be brief.\n\n"
        + records_text(target_tokens, seed)
    )


def test_prefix_cache_hit(server):
    system = _handbook(6000, seed=77)
    q1 = "Which city appears in Record 00042? Reply with the city name only."
    q2 = "How many crates did the warehouse receive in Record 00100? Reply with the number only."

    async def go(s):
        out = {}
        await wait_idle(s)
        for name, q in (("A", q1), ("B", q1), ("C", q2)):
            m0 = await metrics(s)
            r = await stream(s, CHAT, chat_body(q, system=system, max_tokens=48))
            out[name] = dict(result=slim(r), metrics=mdelta(await metrics(s), m0))
        return out

    res = run(_session_do(go))
    A, B, C = (res[k]["result"] for k in "ABC")
    plen = A["prompt_tokens"]
    summary = dict(
        prompt_tokens=plen,
        ttft_s={k: round(res[k]["result"]["ttft_s"], 3) for k in "ABC"},
        prefix_hits={k: res[k]["metrics"]["prefix_cache_hits"] for k in "ABC"},
        answers={k: answer_of(res[k]["result"]) for k in "ABC"},
        b_equals_a=A["token_ids"] == B["token_ids"],
    )
    save("prefix_cache_hit", dict(summary=summary, runs=res))
    assert all(r["status"] == 200 and not r["error"] for r in (A, B, C))
    assert summary["b_equals_a"], (A["text"], B["text"])
    if FEAT["prefix"]:
        full = (plen - 1) // BLOCK * BLOCK
        assert res["B"]["metrics"]["prefix_cache_hits"] >= full - BLOCK, summary
        assert res["C"]["metrics"]["prefix_cache_hits"] >= plen - 200, summary  # the system prompt (+ template)
        assert B["ttft_s"] < 0.35 * A["ttft_s"], summary["ttft_s"]
        assert C["ttft_s"] < 0.5 * A["ttft_s"], summary["ttft_s"]
    else:
        assert res["B"]["metrics"]["prefix_cache_hits"] == 0


# ======================================================================================================================
# 5. multi-turn: a prefix hit on DECODE-written blocks (needs KV-R)
# ======================================================================================================================
def test_multi_turn_decode_written_hit(server):
    tok = tokenizer()
    system = _handbook(2000, seed=91)
    p1 = chat_ids(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "Describe the first ten records in your own words, one line each."},
        ]
    )
    turn2 = tok.encode(
        "<|endofturn|><|startofturn|><|user|>Now list only the cities of those ten records, comma separated."
        "<|endofturn|><|startofturn|><|assistant|><think></think>",
        add_special_tokens=False,
    )

    def comp(prompt, **kw):
        return dict(model=MODEL, prompt=prompt, temperature=0.0, return_token_ids=True, skip_special_tokens=False,
                    **kw)  # fmt: skip

    async def go(s):
        out = {}
        await wait_idle(s)
        m0 = await metrics(s)
        t1 = await stream(s, COMP, comp(p1, max_tokens=192, ignore_eos=True))
        out["turn1"] = dict(result=slim(t1), metrics=mdelta(await metrics(s), m0))
        g1 = list(t1["token_ids"])
        p2 = p1 + g1 + turn2
        # A filler request is admitted first and takes persistent-batch row 0 (where turn 1 ran), so turn 2 takes the
        # next row; vllm-tt-plugin _alloc_prefill_state_slots maps a row to the same state slot when it is free, and
        # the bridge's LaneMap deals slots round-robin over the DP rows (slot 0 -> lane 0, slot 1 -> lane 8).
        gen0 = (await metrics(s)).get("vllm:generation_tokens_total", 0.0)
        filler_task = asyncio.ensure_future(
            stream(s, COMP, comp("Count upwards: 1, 2, 3,", max_tokens=320, ignore_eos=True))
        )
        deadline = time.perf_counter() + 120
        while not filler_task.done() and time.perf_counter() < deadline:  # until the filler is decoding
            await asyncio.sleep(0.25)
            if (await metrics(s)).get("vllm:generation_tokens_total", 0.0) >= gen0 + 3:
                break
        m1 = await metrics(s)
        hit = await stream(s, COMP, comp(p2, max_tokens=64))
        out["turn2_hit"] = dict(result=slim(hit), metrics=mdelta(await metrics(s), m1))
        m2 = await metrics(s)
        cold = await stream(s, COMP, comp(p2, max_tokens=64, cache_salt=f"cold-{uuid.uuid4().hex}"))
        out["turn2_cold"] = dict(result=slim(cold), metrics=mdelta(await metrics(s), m2))
        out["filler"] = slim(await filler_task)
        out["lens"] = dict(p1=len(p1), g1=len(g1), p2=len(p2))
        return out

    res = run(_session_do(go))
    hit, cold = res["turn2_hit"]["result"], res["turn2_cold"]["result"]
    n1, g1 = res["lens"]["p1"], res["lens"]["g1"]
    decode_written_blocks = max(0, ((n1 + g1 - 1) // BLOCK) - (-(-n1 // BLOCK)))  # full blocks of generated tokens
    summary = dict(
        lens=res["lens"],
        hits=dict(
            turn2_hit=res["turn2_hit"]["metrics"]["prefix_cache_hits"],
            turn2_cold=res["turn2_cold"]["metrics"]["prefix_cache_hits"],
        ),  # fmt: skip
        decode_written_full_blocks=decode_written_blocks,
        ttft_s=dict(hit=hit["ttft_s"], cold=cold["ttft_s"]),
        identical=hit["token_ids"] == cold["token_ids"],
        divergence=first_divergence(hit["token_ids"], cold["token_ids"]),
        hit_text=hit["text"][:300],
        cold_text=cold["text"][:300],
    )
    save("multi_turn_hit", dict(summary=summary, runs=res))
    assert all(r["status"] == 200 and not r["error"] for r in (res["turn1"]["result"], hit, cold, res["filler"]))
    assert res["turn1"]["result"]["completion_tokens"] == 192
    if FEAT["prefix"]:
        hits = res["turn2_hit"]["metrics"]["prefix_cache_hits"]
        assert hits >= (n1 + g1) // BLOCK * BLOCK - BLOCK, summary  # the hit reaches into the decode-written blocks
        assert hits > -(-n1 // BLOCK) * BLOCK, summary
        assert res["turn2_cold"]["metrics"]["prefix_cache_hits"] == 0, summary
        assert decode_written_blocks >= 2
    assert summary["identical"], summary


# ======================================================================================================================
# 6. a ~30K-token prompt, chunked, while another request decodes
# ======================================================================================================================
def test_long_prompt_chunked(server):
    text, answer = needle_prompt(30000, seed=31337, depth=0.43)

    async def go(s):
        out = {}
        await wait_idle(s)
        ticker = asyncio.ensure_future(
            stream(s, COMP, dict(model=MODEL, prompt="Count upwards: 1, 2, 3,", max_tokens=800,
                                 temperature=0.0, ignore_eos=True, return_token_ids=True))
        )  # fmt: skip
        await asyncio.sleep(3.0)  # the ticker is decoding
        m0 = await metrics(s)
        t_long = time.perf_counter()
        r1 = await stream(s, CHAT, chat_body(text, max_tokens=48))
        out["first"] = dict(result=slim(r1), metrics=mdelta(await metrics(s), m0), t_send=r1["t_send"],
                            t_first=r1["t_first"])  # fmt: skip
        m1 = await metrics(s)
        r2 = await stream(s, CHAT, chat_body(text, max_tokens=48))
        out["again"] = dict(result=slim(r2), metrics=mdelta(await metrics(s), m1))
        tk = await ticker
        out["ticker"] = slim(tk)
        out["ticker_max_gap_during_prefill_s"] = max_gap(tk, t_long, r1["t_first"])
        out["ticker_max_gap_s"] = max_gap(tk)
        return out

    res = run(_session_do(go))
    r1, r2 = res["first"]["result"], res["again"]["result"]
    summary = dict(
        prompt_tokens=r1["prompt_tokens"],
        expected=answer,
        answers=[answer_of(r1), answer_of(r2)],
        correct=[answer in r1["text"], answer in r2["text"]],
        ttft_s=[r1["ttft_s"], r2["ttft_s"]],
        prefix_hits_again=res["again"]["metrics"]["prefix_cache_hits"],
        ticker_max_gap_during_prefill_s=res["ticker_max_gap_during_prefill_s"],
        ticker_tpot_ms=1e3 * res["ticker"]["tpot_s"] if res["ticker"]["tpot_s"] else None,
        identical=r1["token_ids"] == r2["token_ids"],
    )
    save("long_prompt", dict(summary=summary, runs=res))
    assert r1["status"] == 200 and not r1["error"] and r2["status"] == 200 and not r2["error"]
    assert r1["prompt_tokens"] > 3 * BUDGET, r1["prompt_tokens"]
    assert all(summary["correct"]), summary
    assert summary["identical"], summary
    if FEAT["chunked"]:
        # chunked prefill interleaves decode steps between the ~8K chunks: the stall is one chunk, not the prompt
        gap = summary["ticker_max_gap_during_prefill_s"]
        assert gap is not None and gap < 0.5 * r1["ttft_s"], summary
    if FEAT["prefix"]:
        assert res["again"]["metrics"]["prefix_cache_hits"] >= r1["prompt_tokens"] - 2 * BLOCK, summary
        assert r2["ttft_s"] < 0.2 * r1["ttft_s"], summary


# ======================================================================================================================
# 7. 32 concurrent greedy + sampled requests (PS-1)
# ======================================================================================================================
LENGTHS = [40, 80, 150, 300, 600, 1000, 1500, 2000, 3000, 4000, 6000, 8000]


def _greedy_cases() -> List[Dict[str, Any]]:
    rng = random.Random(2026)
    cases = []
    for i in range(16):
        if i % 4 == 3:
            a, b = rng.randint(100, 999), rng.randint(100, 999)
            text, ans = f"What is {a} + {b}? Show the calculation briefly, then give the result.", str(a + b)
            kind, mt = "arith", 96
        else:
            text, ans = needle_prompt(
                max(LENGTHS[i % len(LENGTHS)], 120), seed=500 + i, depth=rng.choice([0.1, 0.5, 0.9])
            )
            kind, mt = "needle", 32
        cases.append(dict(i=i, kind=kind, text=text, answer=ans, max_tokens=mt))
    return cases


def _sampled_cases() -> List[Dict[str, Any]]:
    out = []
    for i in range(16):
        mode = dict(temperature=1.0, top_p=0.95) if i % 2 == 0 else dict(temperature=0.6, top_p=0.95)
        out.append(dict(i=i, text=QUESTIONS[i], samp=dict(mode, seed=7000 + i), max_tokens=320))
    return out


def _correct(kind: str, expected: str, r: Dict[str, Any]) -> bool:
    ans = answer_of(r)
    return (expected in ans.replace(",", "")) if kind == "arith" else (expected in ans)


async def _greedy_case(s, c, delay=0.0):
    await asyncio.sleep(delay)
    r = await stream(s, CHAT, chat_body(c["text"], max_tokens=c["max_tokens"]))
    out = slim(r)
    out.update(i=c["i"], kind=c["kind"], expected=c["answer"], t_first=r["t_first"], t_last=r["t_last"],
               correct=_correct(c["kind"], c["answer"], r))  # fmt: skip
    return out


async def _sampled_case(s, c, delay=0.0):
    await asyncio.sleep(delay)
    r = await stream(s, CHAT, chat_body(c["text"], max_tokens=c["max_tokens"], ignore_eos=True, **c["samp"]))
    out = slim(r)
    out.update(i=c["i"], samp=c["samp"], t_first=r["t_first"], t_last=r["t_last"], t_send=r["t_send"])
    return out


def test_mixed_concurrency_ps1(server):
    g_cases, s_cases = _greedy_cases(), _sampled_cases()

    async def go(s):
        await wait_idle(s)
        m0 = await metrics(s)
        t0 = time.perf_counter()
        st = [asyncio.ensure_future(_sampled_case(s, c, 0.01 * k)) for k, c in enumerate(s_cases)]
        gt = [asyncio.ensure_future(_greedy_case(s, c, 1.0 + 0.01 * k)) for k, c in enumerate(g_cases)]
        greedy = await asyncio.gather(*gt)
        m1 = await metrics(s)  # every greedy request is done; the sampled ones should still be live
        t1 = time.perf_counter()
        sampled = await asyncio.gather(*st)
        m2 = await metrics(s)
        t2 = time.perf_counter()
        # phase B: the same greedy requests with no sampled request live (speculation resumes; prefix hits)
        greedy_b = await asyncio.gather(*[_greedy_case(s, c, 0.01 * k) for k, c in enumerate(g_cases)])
        m3 = await metrics(s)
        # the greedy versions of 4 sampled prompts (argmax-collapse check)
        coll = await asyncio.gather(*[
            stream(s, CHAT, chat_body(c["text"], max_tokens=c["max_tokens"], ignore_eos=True)) for c in s_cases[:4]
        ])  # fmt: skip
        return dict(greedy=list(greedy), sampled=list(sampled), greedy_b=list(greedy_b),
                    collapse=[slim(r) for r in coll], m_a=mdelta(m1, m0), m_tail=mdelta(m2, m1), m_b=mdelta(m3, m2),
                    wall=dict(a=t1 - t0, sampled=t2 - t0, b=time.perf_counter() - t2))  # fmt: skip

    res = run(_session_do(go))
    greedy, sampled, greedy_b = res["greedy"], res["sampled"], res["greedy_b"]
    live_throughout = min(x["t_last"] for x in sampled) > max(x["t_last"] for x in greedy) and min(
        x["t_send"] for x in sampled
    ) < min(x["t_first"] for x in greedy)
    same_b = [a["token_ids"] == b["token_ids"] for a, b in zip(greedy, greedy_b)]
    collapsed = [r["token_ids"] == c["token_ids"] for r, c in zip(res["collapse"], sampled[:4])]
    text = server_log_text()
    canary = None if text is None else len(re.findall(r"not speculable", text))
    summary = dict(
        status_ok=sum(x["status"] == 200 and not x["error"] for x in greedy + sampled + greedy_b),
        greedy_correct=f"{sum(x['correct'] for x in greedy)}/16",
        greedy_b_correct=f"{sum(x['correct'] for x in greedy_b)}/16",
        greedy_b_identical=f"{sum(same_b)}/16",
        greedy_b_different=[x["i"] for x, ok in zip(greedy, same_b) if not ok],
        sampled_tokens=[x["completion_tokens"] for x in sampled],
        sampled_live_while_greedy_ran=live_throughout,
        metrics_phase_a=res["m_a"],
        metrics_phase_b=res["m_b"],
        sampled_equal_to_greedy=f"{sum(collapsed)}/4",
        ps1_canary_lines=canary,
        ttft_greedy=stats([x["ttft_s"] for x in greedy]),
        ttft_sampled=stats([x["ttft_s"] for x in sampled]),
        wall_s=res["wall"],
    )
    save("mixed_ps1", dict(summary=summary, runs=res))
    assert summary["status_ok"] == 48
    assert sum(x["correct"] for x in greedy) == 16, [
        (x["i"], x["expected"], x["text"][-120:]) for x in greedy if not x["correct"]
    ]
    assert sum(x["correct"] for x in greedy_b) == 16
    assert all(x["completion_tokens"] == 320 for x in sampled)
    assert sum(collapsed) == 0, "a sampled request produced exactly its greedy text (argmax collapse)"
    if not FEAT["packed"]:
        # phase A prefills the rows cold, phase B from prefix hits: per-row launches give the same tokens. A packed
        # launch runs phase A's rows in packed passes whose grouping depends on arrival timing: recorded (both phases'
        # answers are asserted correct above)
        assert all(same_b), summary["greedy_b_different"]
    if SPEC:
        assert live_throughout, "the sampled requests did not outlive the greedy ones: the PS-1 window is not clean"
        assert (
            res["m_a"]["spec_decode_num_drafts"] == 0
        ), f"drafts scheduled while sampled requests were live: {res['m_a']}"
        assert res["m_b"]["spec_decode_num_drafts"] > 0, res["m_b"]
        if canary is not None:
            assert canary == 0, "the plugin logged a verify step that held a non-speculable request (PS-1 broken)"
    if FEAT["prefix"]:
        assert res["m_b"]["prefix_cache_hits"] > 0


# ======================================================================================================================
# 7b. requests a speculating launch cannot serve faithfully are refused (features design §1.1 variants, §5.4)
# ======================================================================================================================
def test_spec_refuses_logprobs(server):
    """``logprobs`` need logits that a verify step never returns (``argmax_ids`` mode): the plugin refuses such a
    request with HTTP 400 on a speculating launch instead of answering without them; other launches serve it."""

    async def go(s):
        body = chat_body("Name one primary color.", max_tokens=4, logprobs=True, top_logprobs=2)
        body["stream"] = False
        async with s.post(BASE + CHAT, json=body) as r:
            return r.status, (await r.text())[:2000]

    status, text = run(_session_do(go))
    save("spec_refusals", dict(summary=dict(status=status, body=text[:400])))
    if SPEC:
        assert status == 400 and "logprobs" in text and "peculative" in text, (status, text)
    else:
        assert status == 200 and '"top_logprobs"' in text, (status, text)


# ======================================================================================================================
# 8. TTFT under a burst (FR §3.7; E2E-P (a) / (b))
# ======================================================================================================================
def _burst_summary(v: Dict[str, Any]) -> Dict[str, Any]:
    """TTFT stats of a natural burst; vLLM runs its first arrival alone, so the rest's first tokens come one decode
    step + one prefill step later: ``second_step_s`` = p50 of the others' TTFT - the first TTFT."""
    t = sorted(r["ttft_s"] for r in v["results"] if r["ttft_s"] is not None)
    rest = stats(t[1:])
    return dict(ttft=stats(t), wall_s=round(v["wall_s"], 2), second_step_s=(rest["p50"] - t[0]) if t[1:] else None,
                prompt_tokens=stats([r["prompt_tokens"] for r in v["results"]]),
                prefix_hits=v["metrics"]["prefix_cache_hits"], server_prefill_s_mean=v["metrics"]["prefill_s_mean"],
                status_ok=sum(r["status"] == 200 for r in v["results"]), loadavg_1m=v["loadavg"])  # fmt: skip


def test_burst_ttft(server):
    shared = _handbook(2000, seed=123)

    async def go(s):
        out = {}
        for name in ("unique", "shared_2k"):
            await wait_idle(s)
            await asyncio.sleep(1.0)
            m0 = await metrics(s)
            load = os.getloadavg()[0]
            t0 = time.perf_counter()
            salt = uuid.uuid4().hex[:8]
            if name == "unique":
                bodies = [chat_body(f"[{uuid.uuid4().hex[:8]}] {QUESTIONS[i]}", max_tokens=8) for i in range(32)]
            else:
                bodies = [chat_body(f"{QUESTIONS[i]} (ref {salt})", system=shared, max_tokens=8) for i in range(32)]
            rs = await asyncio.gather(*[stream(s, CHAT, b) for b in bodies])
            out[name] = dict(results=[slim(r) for r in rs], wall_s=time.perf_counter() - t0,
                             metrics=mdelta(await metrics(s), m0), loadavg=round(load, 1))  # fmt: skip
        return out

    res = run(_session_do(go))
    summary = {k: _burst_summary(v) for k, v in res.items()}
    save("burst_ttft", dict(summary=summary, runs=res))
    for v in res.values():
        assert all(r["status"] == 200 and not r["error"] for r in v["results"])
    if FEAT["prefix"]:
        assert res["shared_2k"]["metrics"]["prefix_cache_hits"] > 20 * 1900, summary
    if FEAT["packed"]:  # E2E-P bars (P5_T64_DESIGN §6.2)
        a, b = summary["unique"], summary["shared_2k"]
        assert a["ttft"]["mean"] <= BAR_BURST_A_TTFT_MEAN_S, a
        assert b["ttft"]["mean"] <= BAR_BURST_B_TTFT_MEAN_S, b
        # the second step = one decode step (~0.1 s) + the packed / pk1 prefill step
        assert a["second_step_s"] is not None and a["second_step_s"] <= BAR_PACKED_STEP_A_S + 0.25, a
        assert b["second_step_s"] is not None and b["second_step_s"] <= BAR_PK1_STEP_B_S + 0.25, b


# ======================================================================================================================
# 9. decode throughput at c = 1 / 8 / 16 / 20 / 24 / 32 (G-serve)
# ======================================================================================================================
def _steady(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate tokens/s over the window where every request of the level is decoding (after the last first token,
    before the first last token), counting streamed chunks' tokens (1 token per chunk; 2 after an accepted draft)."""
    lo = max(r["t_first"] for r in results)
    hi = min(r["t_last"] for r in results)
    if hi <= lo:
        return dict(window_s=0.0)
    n = 0
    for r in results:
        ts = [r["t_send"] + c for c in r["chunk_times"]]
        n += sum(k for t, k in zip(ts, r["chunk_ntok"]) if lo < t <= hi)
    return dict(window_s=hi - lo, tokens=n, tok_s=n / (hi - lo))


def _tput_levels() -> List[int]:
    return [int(x) for x in os.environ.get("MOTIF3_E2E_TPUT_LEVELS", "1,8,16,20,24,32").split(",")]


def test_decode_throughput(server):
    """Greedy decode at each level, gated (the level's requests prefill in one step, so every launch decodes the same
    prompts from the same KV): per-user TPOT, the steady-window aggregate tok/s, drafts and acceptance. Without MTP no
    draft; with MTP drafts at c <= 16 (idle lanes), and on a T64 launch (``auto`` / ``wide``) at every level."""
    levels = _tput_levels()
    max_tokens = int(os.environ.get("MOTIF3_E2E_TPUT_TOKENS", "256"))
    thinking = None if os.environ.get("MOTIF3_E2E_TPUT_THINKING", "on").strip().lower() == "on" else False

    async def go(s):
        out = {}
        for c in levels:
            jobs = [(CHAT, chat_body(QUESTIONS[i % len(QUESTIONS)], max_tokens=max_tokens, thinking=thinking,
                                     ignore_eos=True, cache_salt=uuid.uuid4().hex)) for i in range(c)]  # fmt: skip
            load0 = os.getloadavg()  # host CPU contention slows the server's host side (sampling, input build)
            t0 = time.perf_counter()
            g = await gated(s, jobs)
            out[str(c)] = dict(g=g, jobs=jobs, wall_s=time.perf_counter() - t0, loadavg=[load0, os.getloadavg()])
        return out

    res = run(_session_do(go))
    summary, runs = {}, {}
    for c, v in res.items():
        g = v["g"]
        rs = g["results"]
        md = mdelta(g["m_end"], g["m_gate"])
        tp = [tpot(r) for r in rs]
        summary[c] = dict(
            status_ok=sum(r["status"] == 200 and not r["error"] for r in rs),
            gate_ok=g["gate"]["ok"],
            tokens=sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in rs),
            tpot_ms=stats([1e3 * t for t in tp if t]),
            per_user_tok_s=stats([1.0 / t for t in tp if t]),
            steady=_steady(rs),
            wall_s=round(v["wall_s"], 2),
            first_token_after_gate_s=step_stats(g)["first_token_after_gate_s"],
            acceptance=md.get("acceptance"),
            drafts=md["spec_decode_num_drafts"],
            loadavg_1m=[round(x[0], 1) for x in v["loadavg"]],
            multi_token_chunks=sum(sum(1 for k in r["chunk_ntok"] if k > 1) for r in rs),
        )
        runs[c] = gated_record(g, v["jobs"])
    save("decode_throughput", dict(summary=summary, runs=runs, max_tokens=max_tokens, thinking=thinking))
    for c, v in summary.items():
        assert v["status_ok"] == int(c)
        assert v["tokens"] == int(c) * max_tokens
        if SPEC and (int(c) <= 16 or WIDE):  # packed verify: no idle lane at 32 live lanes (by design)
            assert v["drafts"] > 0 and (v["acceptance"] or 0) > 0.4, (c, v)
        if not SPEC:
            assert v["drafts"] == 0, (c, v)


# ======================================================================================================================
# 10. P5: gated bursts (one-step and multi-step), packed prefill's shapes through vLLM
# ======================================================================================================================
MT16_TURN1_TOKENS = 700  # every session's turn 1 has exactly this many tokens: turn 2 resumes at one common start


def _mt16_sessions() -> List[Dict[str, Any]]:
    """16 sessions whose turn-1 prompts are exactly ``MT16_TURN1_TOKENS`` token ids (raw completions: a log with a
    passphrase near the top, then the question with the answer's opening quote); turn 2 = turn 1 + a fixed reply (the
    right passphrase) + a second question. Turn 2 hits turn 1's 10 full blocks (640 tokens, a multiple of A = 128),
    so all 16 resume at start 640 with distinct tails: one pk1 pass (S 128, B 16, ``distinct``)."""
    tok = tokenizer()
    out = []
    for k in range(16):
        vault = 800 + k
        rng = random.Random(vault)
        code = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
        recs = [_record(rng, i) for i in range(40)]
        recs.insert(3, f"IMPORTANT NOTE: the secret passphrase for vault {vault} is '{code}'.")
        ctx = tok.encode("Below is a log of warehouse records.\n\n" + "\n".join(recs), add_special_tokens=False)
        q1 = tok.encode(
            f"\n\nQuestion: according to the records above, what is the secret passphrase for vault {vault}?\nAnswer: "
            f"The secret passphrase is '",
            add_special_tokens=False,
        )
        ids1 = [bos_id()] + ctx[: MT16_TURN1_TOKENS - 1 - len(q1)] + q1
        assert len(ids1) == MT16_TURN1_TOKENS and len(ctx) > MT16_TURN1_TOKENS, (len(ids1), len(ctx))
        reply = tok.encode(f"{code}'.\n\nQuestion: which vault number was that passphrase for?\nAnswer: Vault",
                           add_special_tokens=False)  # fmt: skip
        out.append(dict(vault=vault, code=code, ids1=ids1, ids2=ids1 + reply))
    return out


def _mixed32_jobs(salt: str) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Dict[str, Any]]]:
    """The SERVING_SMOKE §5 32-request mixed burst (``logs/serve/clients/concurrency_test.py``: 24 needle prompts of
    40-8000 tokens + 8 arithmetic questions), greedy except 8 seeded sampled rows (T 1.0 / top-p 0.95)."""
    rng = random.Random(2027)
    jobs, cases = [], []
    for i in range(32):
        if i % 4 == 3:
            a, b = rng.randint(100, 999), rng.randint(100, 999)
            text, ans = f"What is {a} + {b}? Show the calculation briefly, then give the result.", str(a + b)
            kind, mt = "arith", 96
        else:
            text, ans = needle_prompt(max(LENGTHS[i % len(LENGTHS)], 120), seed=900 + i,
                                      depth=rng.choice([0.1, 0.3, 0.5, 0.7, 0.9]))  # fmt: skip
            kind, mt = "needle", 32
        samp = dict(temperature=1.0, top_p=0.95, seed=9100 + i) if i % 4 == 1 else dict(temperature=0.0, top_p=1.0)
        jobs.append((CHAT, chat_body(text, max_tokens=mt, cache_salt=salt, **samp)))
        cases.append(dict(i=i, kind=kind, answer=ans, sampled=i % 4 == 1))
    return jobs, cases


def test_p5_gated_bursts(server):
    """Gated bursts through the whole stack (E2E-P): 32 short prompts (pk0, one step) and the same batch again (repeat
    determinism: bitwise), 32 sharing a 2K prefix in one step (row 0 cold, 31 same-step hits: solo sp0 2048 + a pk1
    pass), 8 identical 6K prompts (same-step hits behind their writer's chunks), the SERVING_SMOKE 32-request mixed
    burst over several steps, and a 16-session turn 2 at one common start (pk1, distinct tails) after its turn 1 (pk0
    S 1024). Asserts answers, prefix hits and the repeat; records the step times; ``test_92`` compares the tokens
    with the reference launch."""
    tag = uuid.uuid4().hex[:8]
    shared = _handbook(2000, seed=123)
    text6k, code6k = needle_prompt(6000, seed=6006, depth=0.6)
    mixed_jobs, mixed_cases = _mixed32_jobs(f"mixed32-{tag}")
    sessions = _mt16_sessions()

    def short32(salt):
        return [(CHAT, chat_body(QUESTIONS[i], max_tokens=64, cache_salt=salt)) for i in range(32)]

    workloads = [
        ("short32", short32(f"short32-{tag}"), 0.02),
        ("short32_rep", short32(f"short32rep-{tag}"), 0.02),
        ("shared2k", [(CHAT, chat_body(QUESTIONS[i], system=shared, max_tokens=64, cache_salt=f"shared2k-{tag}"))
                      for i in range(32)], 0.02),
        ("same6k", [(CHAT, chat_body(text6k, max_tokens=32, cache_salt=f"same6k-{tag}")) for _ in range(8)], 0.03),
        ("mixed32", mixed_jobs, 0.04),
        ("mt16_t1", [(COMP, comp_body(x["ids1"], max_tokens=16, cache_salt=f"mt16-{tag}")) for x in sessions], 0.02),
        ("mt16_t2", [(COMP, comp_body(x["ids2"], max_tokens=8, cache_salt=f"mt16-{tag}")) for x in sessions], 0.02),
    ]  # fmt: skip

    async def go(s):
        out = {}
        for name, jobs, stagger in workloads:
            g = await gated(s, jobs, stagger=stagger)
            out[name] = gated_record(g, jobs)
            log(f"p5 gated {name}: {json.dumps(out[name]['step'])[:400]}")
        return out

    runs = run(_session_do(go))
    R = {k: v["results"] for k, v in runs.items()}
    rep_same = [a["token_ids"] == b["token_ids"] for a, b in zip(R["short32"], R["short32_rep"])]
    six = R["same6k"]
    readers_identical = all(x["token_ids"] == six[1]["token_ids"] for x in six[1:])
    mixed_ok = [_correct(c["kind"], c["answer"], r) for c, r in zip(mixed_cases, R["mixed32"])]
    mt1 = [x["code"] in r["text"] for x, r in zip(sessions, R["mt16_t1"])]
    mt2 = [str(x["vault"]) in r["text"] for x, r in zip(sessions, R["mt16_t2"])]
    summary = dict(
        gates_ok={k: v["gate"]["ok"] for k, v in runs.items()},
        status_ok={k: sum(r["status"] == 200 and not r["error"] for r in v) for k, v in R.items()},
        steps={k: v["step"] for k, v in runs.items()},
        prompt_tokens={k: sum(r["prompt_tokens"] or 0 for r in v) for k, v in R.items()},
        prefix_hits={k: v["metrics"]["prefix_cache_hits"] for k, v in runs.items()},
        repeat_identical=f"{sum(rep_same)}/32",
        same6k=dict(correct=sum(code6k in r["text"] for r in six), readers_identical=readers_identical,
                    writer_equals_readers=six[0]["token_ids"] == six[1]["token_ids"]),  # fmt: skip
        mixed32=dict(greedy_correct=f"{sum(ok for ok, c in zip(mixed_ok, mixed_cases) if not c['sampled'])}/24",
                     sampled_correct=f"{sum(ok for ok, c in zip(mixed_ok, mixed_cases) if c['sampled'])}/8"),
        mt16=dict(turn1_correct=f"{sum(mt1)}/16", turn2_correct=f"{sum(mt2)}/16"),
    )  # fmt: skip
    save("p5_gated", dict(summary=summary, runs=runs))
    for k, v in R.items():
        assert all(r["status"] == 200 and not r["error"] for r in v), (k, [r["error"] for r in v if r["error"]])
    assert all(summary["gates_ok"].values()), summary["gates_ok"]
    assert all(
        rep_same
    ), f"the same gated batch twice gave other tokens: {[i for i, ok in enumerate(rep_same) if not ok]}"
    assert summary["same6k"]["correct"] == 8 and readers_identical, summary["same6k"]
    assert all(ok for ok, c in zip(mixed_ok, mixed_cases) if not c["sampled"]), summary["mixed32"]
    assert sum(ok for ok, c in zip(mixed_ok, mixed_cases) if c["sampled"]) >= 6, summary["mixed32"]
    assert all(mt1) and all(mt2), summary["mt16"]
    if FEAT["prefix"]:
        assert runs["shared2k"]["metrics"]["prefix_cache_hits"] >= 31 * 2048, summary["prefix_hits"]
        assert runs["same6k"]["metrics"]["prefix_cache_hits"] >= 7 * 5888, summary["prefix_hits"]
        assert runs["mt16_t2"]["metrics"]["prefix_cache_hits"] >= 16 * 640, summary["prefix_hits"]
    if FEAT["packed"]:  # one pk0 pass (S 64, B 32: T 2048) for 32 short prompts: the E2E-P step bar
        step = runs["short32"]["step"]["server_prefill_s_mean"]
        assert step is not None and step <= BAR_PACKED_STEP_A_S, runs["short32"]["step"]


# ======================================================================================================================
# 11. P5: decode stall while a burst prefills
# ======================================================================================================================
def test_p5_decode_stall(server):
    """(1) one request decoding while 31 short prompts arrive; (2) 16 decoding while 16 arrive. The decoding requests'
    worst inter-token gap from the burst's submission to its last first token is the stall one prefill step costs
    (today ~0.65 s x rows; packed: the packed step + one decode step; E2E-P bar 2.5 s)."""

    async def phase(s, tickers: List[Tuple[str, Dict[str, Any]]], n_burst: int, q0: int) -> Dict[str, Any]:
        await wait_idle(s)
        gen0 = (await metrics(s)).get("vllm:generation_tokens_total", 0.0)
        tk = [asyncio.ensure_future(stream(s, p, b)) for p, b in tickers]
        deadline = time.perf_counter() + 300
        while time.perf_counter() < deadline:  # every ticker is decoding (~6 tokens each)
            await asyncio.sleep(0.25)
            if (await metrics(s)).get("vllm:generation_tokens_total", 0.0) >= gen0 + 6 * len(tickers):
                break
        await asyncio.sleep(1.0)
        t_b = time.perf_counter()
        burst = await asyncio.gather(*[
            stream(s, CHAT, chat_body(f"[{uuid.uuid4().hex[:8]}] {QUESTIONS[(q0 + i) % 32]}", max_tokens=8))
            for i in range(n_burst)
        ])  # fmt: skip
        firsts = [r["t_first"] for r in burst if r["t_first"]]
        t_e = max(firsts) if firsts else time.perf_counter()
        tres = await asyncio.gather(*tk)
        gaps = [max_gap(r, t_b, t_e) for r in tres]
        return dict(burst=[slim(r) for r in burst], tickers=[slim(r) for r in tres],
                    worst_gap_s=max(g for g in gaps if g is not None) if any(g is not None for g in gaps) else None,
                    gaps_s=gaps, burst_ttft=stats([r["ttft_s"] for r in burst]), window_s=t_e - t_b)  # fmt: skip

    async def go(s):
        one = [(COMP, comp_body("Count upwards: 1, 2, 3,", max_tokens=200, ignore_eos=True))]
        sixteen = [(CHAT, chat_body(QUESTIONS[16 + i], max_tokens=256, thinking=None, ignore_eos=True,
                                    cache_salt=uuid.uuid4().hex)) for i in range(16)]  # fmt: skip
        return dict(one_plus_31=await phase(s, one, 31, 0), sixteen_plus_16=await phase(s, sixteen, 16, 0))

    res = run(_session_do(go))
    summary = {k: dict(worst_gap_s=v["worst_gap_s"], burst_ttft=v["burst_ttft"], window_s=round(v["window_s"], 2),
                       status_ok=sum(r["status"] == 200 and not r["error"] for r in v["burst"] + v["tickers"]))
               for k, v in res.items()}  # fmt: skip
    save("p5_decode_stall", dict(summary=summary, runs=res))
    for k, v in res.items():
        assert all(r["status"] == 200 and not r["error"] for r in v["burst"] + v["tickers"]), k
        assert v["worst_gap_s"] is not None, (k, "a ticker finished before the burst")
    if FEAT["packed"]:
        for k, v in summary.items():
            assert v["worst_gap_s"] <= BAR_STALL_S, (k, v)


# ======================================================================================================================
# 12. P5: a long prompt's chunks next to a short burst
# ======================================================================================================================
def test_p5_long_beside_burst(server):
    """A ~30K-token needle prompt (4 vLLM chunks of <= the budget) and, 2 s later, 16 short prompts: the long row's
    chunks run solo (the stall of one ~8K chunk), the short rows pack when a step has room for them."""
    text, answer = needle_prompt(30000, seed=31338, depth=0.61)
    tag = uuid.uuid4().hex[:8]

    async def go(s):
        await wait_idle(s)
        m0 = await metrics(s)
        long_t = asyncio.ensure_future(stream(s, CHAT, chat_body(text, max_tokens=48, cache_salt=f"long-{tag}")))
        await asyncio.sleep(2.0)
        burst = await asyncio.gather(*[
            stream(s, CHAT, chat_body(QUESTIONS[i], max_tokens=24, cache_salt=f"lb-{tag}")) for i in range(16)
        ])  # fmt: skip
        r = await long_t
        return dict(
            long=slim(r),
            burst=[slim(x) for x in burst],
            metrics=mdelta(await metrics(s), m0),
            requests=[chat_body(QUESTIONS[i], max_tokens=24, cache_salt=f"lb-{tag}") for i in range(16)],
        )

    res = run(_session_do(go))
    summary = dict(
        long_correct=answer in res["long"]["text"],
        long_ttft_s=res["long"]["ttft_s"],
        long_prompt_tokens=res["long"]["prompt_tokens"],
        burst_ttft=stats([x["ttft_s"] for x in res["burst"]]),
        status_ok=sum(x["status"] == 200 and not x["error"] for x in [res["long"]] + res["burst"]),
    )
    save("p5_long_beside_burst", dict(summary=summary, runs=res))
    assert summary["status_ok"] == 17, summary
    assert summary["long_correct"], (answer, res["long"]["text"][-200:])
    assert res["long"]["prompt_tokens"] > 3 * BUDGET


# ======================================================================================================================
# 13. T64 + PS-1: sampled and greedy requests at c = 32
# ======================================================================================================================
def test_t64_ps1_mix(server):
    """8 seeded sampled requests (96 tokens) + 24 greedy ones (thinking on, 320 tokens), gated. PS-1: no draft while
    any sampled request is live. Once they finished, 24 greedy lanes are left: a T64 launch (c* ~19) drafts every lane
    and verifies them in the 64-row trace; ``packed`` verify drafts the 8 idle lanes. ``test_92`` compares the greedy
    and the seeded sampled tokens with the reference launch."""
    tag = uuid.uuid4().hex[:8]
    jobs = []
    for i in range(8):
        samp = dict(temperature=1.0, top_p=0.95) if i % 2 == 0 else dict(temperature=0.6, top_p=0.95)
        jobs.append((CHAT, chat_body(QUESTIONS[24 + i], max_tokens=96, ignore_eos=True, seed=8100 + i,
                                     cache_salt=f"ps1-{tag}", **samp)))  # fmt: skip
    for i in range(24):
        jobs.append((CHAT, chat_body(QUESTIONS[i], max_tokens=320, thinking=None, ignore_eos=True,
                                     cache_salt=f"ps1-{tag}")))  # fmt: skip

    async def go(s):
        return await gated(s, jobs, poll=0.2)

    g = run(_session_do(go))
    rs = g["results"]
    sampled, greedy = rs[:8], rs[8:]
    t_a = max(r["t_last"] for r in sampled if r["t_last"])
    polls = g["polls"]
    d0 = polls[0][1] if polls else 0.0
    in_a = [d for t, d in polls if t <= t_a - 0.3]
    drafts_a = (in_a[-1] - d0) if in_a else 0.0
    md = mdelta(g["m_end"], g["m_gate"])
    drafts_b = md["spec_decode_num_drafts"] - drafts_a

    def multi(r, lo, hi):
        ts = [r["t_send"] + c for c in r["chunk_times"]]
        return sum(1 for t, k in zip(ts, r["chunk_ntok"]) if k > 1 and lo <= t <= hi)

    multi_a = sum(multi(r, 0.0, t_a - 0.3) for r in greedy)
    multi_b = sum(multi(r, t_a + 0.3, float("inf")) for r in greedy)
    text = server_log_text()
    canary = None if text is None else len(re.findall(r"not speculable", text))
    summary = dict(
        gate_ok=g["gate"]["ok"],
        status_ok=sum(r["status"] == 200 and not r["error"] for r in rs),
        tokens=dict(
            sampled=[r["usage"]["completion_tokens"] if r["usage"] else None for r in sampled],
            greedy=sorted({r["usage"]["completion_tokens"] if r["usage"] else None for r in greedy}, key=str),
        ),  # fmt: skip
        window_a_s=round(t_a - g["t_gate"], 2) if g["t_gate"] else None,
        drafts_while_sampled_live=drafts_a,
        drafts_after=drafts_b,
        multi_token_chunks_while_sampled_live=multi_a,
        multi_token_chunks_after=multi_b,
        acceptance=md.get("acceptance"),
        greedy_tpot_ms=stats([1e3 * t for t in (tpot(r) for r in greedy) if t]),
        ps1_canary_lines=canary,
        polls=len(polls),
    )
    save("t64_ps1_mix", dict(summary=summary, runs=dict(mix=gated_record(g, jobs))))
    assert summary["status_ok"] == 32, [r["error"] for r in rs if r["error"]]
    assert all(r["usage"]["completion_tokens"] == 96 for r in sampled)
    assert all(r["usage"]["completion_tokens"] == 320 for r in greedy)
    assert g["gate"]["ok"], g["gate"]
    if SPEC:
        assert len(polls) >= 10 and drafts_a == 0, summary
        assert drafts_b > 0, summary  # speculation resumes once only greedy requests are left
        if WIDE:  # 24 live lanes >= c*: every lane drafts, the drafts that do not fit idle lanes go to T64
            assert multi_b > 0 and (md.get("acceptance") or 0) > 0.4, summary
        if canary is not None:
            assert canary == 0, summary
    else:  # (multi-token chunks can also be two steps' tokens in one SSE event: recorded, not asserted)
        assert md["spec_decode_num_drafts"] == 0, summary


# ======================================================================================================================
# 14. P5: the pass-size floor (the same rows in passes of T = 256 ... 2048)
# ======================================================================================================================
PASS_SIZES = (4, 8, 16, 32)


def test_p5_pass_size_floor(server):
    """The 32 short prompts (greedy, 64 tokens) in gated groups of B = 4 / 8 / 16 / 32 rows. A packed launch runs each
    group as one pk0 pass of T = 64 B rows (256 / 512 / 1024 / 2048), a per-row launch as solo bucket-128 chunks
    whatever B. Per-row: a prompt's tokens must not depend on B. Packed: recorded; the differences between the B show
    the M = T floor of the row-local programs at bucket T, which a packed pass inherits by design (P5_T64_DESIGN §3.6:
    "a row can differ ... between packed and solo"; gate CP-P: packed vs per-row = per-row at bucket T vs per-row)."""
    tag = uuid.uuid4().hex[:8]

    async def go(s):
        out = {}
        for B in PASS_SIZES:
            reqs, res, gates = [], [], []
            for g0 in range(0, 32, B):
                jobs = [(CHAT, chat_body(QUESTIONS[i], max_tokens=64, cache_salt=f"floor{B}-{tag}"))
                        for i in range(g0, g0 + B)]  # fmt: skip
                g = await gated(s, jobs)
                reqs += [b for _, b in jobs]
                res += [slim(r) for r in g["results"]]
                gates.append(g["gate"]["ok"])
            out[str(B)] = dict(requests=reqs, results=res, gates_ok=gates)
        return out

    runs = run(_session_do(go))
    ids = {B: [r["token_ids"] for r in v["results"]] for B, v in runs.items()}
    keys = list(ids)
    between = {}
    for i, x in enumerate(keys):
        for y in keys[i + 1 :]:
            between[f"B{x}_vs_B{y}"] = f"{sum(a == b for a, b in zip(ids[x], ids[y]))}/32"
    ref = load(REF_DIR, "p5_pass_size_floor")
    vs_ref = None
    if ref is not None:
        vs_ref = {f"B{B}": f"{sum(a == b['token_ids'] for a, b in zip(ids[B], ref['runs'][B]['results']))}/32"
                  for B in ids if B in ref["runs"]}  # fmt: skip
    summary = dict(gates_ok=all(all(v["gates_ok"]) for v in runs.values()), identical_between_sizes=between,
                   identical_to_reference=vs_ref, reference_profile=ref.get("profile") if ref else None)  # fmt: skip
    save("p5_pass_size_floor", dict(summary=summary, runs=runs))
    assert all(r["status"] == 200 and not r["error"] for v in runs.values() for r in v["results"])
    assert summary["gates_ok"], summary
    if not FEAT["packed"]:  # a per-row launch prefills every row alone: the batch does not matter
        assert all(v == "32/32" for v in between.values()), between


# ======================================================================================================================
# 15. P5: the logprob replay (both launches' log-probabilities where they diverge)
# ======================================================================================================================
def test_p5_logprob_replay(server):
    """The gated 32 short prompts once more with ``top_logprobs`` 5 (``return_tokens_as_token_ids``). The same pass and
    the same logits (the logprob steps sample on the host), so the tokens must equal ``test_p5_gated_bursts``' device-
    sampled ones; ``test_94`` then reads both launches' log-probabilities where a packed and a per-row launch diverge.
    A speculating launch refuses logprobs: skipped."""
    if SPEC:
        pytest.skip("a speculating launch refuses logprobs (test_spec_refuses_logprobs)")
    tag = uuid.uuid4().hex[:8]
    jobs = [(CHAT, chat_body(QUESTIONS[i], max_tokens=64, cache_salt=f"lpr-{tag}", logprobs=True, top_logprobs=5,
                             return_tokens_as_token_ids=True)) for i in range(32)]  # fmt: skip

    async def go(s):
        return await gated(s, jobs)

    g = run(_session_do(go))
    rec = gated_record(g, jobs)
    mine = load(OUT_DIR, "p5_gated")
    same = None
    if mine is not None:
        same = [a["token_ids"] == b["token_ids"] for a, b in zip(rec["results"], mine["runs"]["short32"]["results"])]
    lp_ok = all(len(r.get("top_logprobs") or []) == len(r["token_ids"]) for r in rec["results"])
    summary = dict(gate_ok=g["gate"]["ok"], logprobs_per_token=lp_ok, step=rec["step"],
                   identical_to_device_sampled=f"{sum(same)}/{len(same)}" if same is not None else None)  # fmt: skip
    save("p5_logprob_replay", dict(summary=summary, runs=dict(short32=rec)))
    assert all(r["status"] == 200 and not r["error"] for r in rec["results"])
    assert g["gate"]["ok"] and lp_ok, summary
    if same is not None:
        assert all(same), summary


# ======================================================================================================================
# comparisons with the reference server's results (no server needed, except for near-tie probes)
# ======================================================================================================================
def _need_both(name: str):
    a, b = load(OUT_DIR, name), load(REF_DIR, name)
    if a is None or b is None:
        pytest.skip(f"{name}: needs MOTIF3_E2E_OUT and MOTIF3_E2E_REFERENCE results")
    return a, b


def _ref_profile() -> Optional[str]:
    for name in ("server_config", "p5_gated", "decode_throughput", "smoke_chats"):
        d = load(REF_DIR, name)
        if d and d.get("profile") in PROFILES:
            return d["profile"]
    return None


def _same_prefill(ref: Optional[str]) -> bool:
    """Both launches prefill a gated burst the same way (the same vLLM chunking and the same per-row or packed path),
    so their greedy tokens must be identical (speculation and the device sampler are lossless for greedy)."""
    return ref is not None and all(FEAT[k] == PROFILES[ref][k] for k in ("chunked", "prefix", "packed"))


def _is_seeded_sampled(body: Dict[str, Any]) -> bool:
    return float(body.get("temperature", 0.0)) > 0 and body.get("seed") is not None


def _pairs(cur: Dict[str, Any], ref: Dict[str, Any], label: str):
    """(label, body, this launch's ids, the reference's ids, sampled?) of a gated workload's requests."""
    out = []
    for i, (body, a, b) in enumerate(zip(cur["requests"], cur["results"], ref["results"])):
        out.append((f"{label}[{i}]", body, list(a["token_ids"]), list(b["token_ids"]), _is_seeded_sampled(body)))
    return out


def _check_pairs(name: str, pairs, exact: bool, sampled_comparable: bool) -> Dict[str, Any]:
    """Greedy pairs: identical (``exact``), or every difference a near-tie (probed on this live server when it is a
    non-speculating launch). Seeded sampled pairs: identical when ``sampled_comparable`` (both launches sample on the
    device and prefill alike), else recorded."""
    greedy = [p for p in pairs if not p[4]]
    sampled = [p for p in pairs if p[4]]
    g_diff = [p for p in greedy if p[2] != p[3]]
    s_diff = [p for p in sampled if p[2] != p[3]]
    probes: List[Optional[Dict[str, Any]]] = [None] * len(g_diff)
    probed = bool(g_diff) and not exact and not SPEC and _server_up()
    if probed:

        async def go(s):
            return [await near_tie_probe(s, body, a, b) for _, body, a, b, _ in g_diff]

        probes = run(_session_do(go))
    rows = [dict(label=lab, divergence=first_divergence(a, b), len=[len(a), len(b)], probe=pr)
            for (lab, _, a, b, _), pr in zip(g_diff, probes)]  # fmt: skip
    out = dict(greedy=len(greedy), greedy_identical=len(greedy) - len(g_diff), greedy_mismatches=rows,
               sampled=len(sampled), sampled_identical=len(sampled) - len(s_diff),
               sampled_mismatches=[dict(label=lab, divergence=first_divergence(a, b)) for lab, _, a, b, _ in s_diff],
               mode="exact" if exact else "floor", probed=probed)  # fmt: skip
    log(f"compare {name}: greedy {out['greedy_identical']}/{out['greedy']} sampled {out['sampled_identical']}/"
        f"{out['sampled']} ({out['mode']}{', probed' if probed else ''})")  # fmt: skip
    return out


def _assert_pairs(name: str, res: Dict[str, Any], exact: bool, sampled_comparable: bool) -> None:
    if exact:
        assert not res["greedy_mismatches"], (name, res["greedy_mismatches"][:4])
        if sampled_comparable:
            assert not res["sampled_mismatches"], (name, res["sampled_mismatches"][:4])
        return
    if res["probed"]:
        bad = [m for m in res["greedy_mismatches"] if not near_tie_ok(m["probe"])]
        assert not bad, (
            name,
            f"greedy differences that are not near-ties (top {NEAR_TIE_RANK}, <= {NEAR_TIE})",
            bad[:4],
        )
    else:  # nothing to probe with: a speculating or stopped server; bound the differences, the probe can be re-run
        assert len(res["greedy_mismatches"]) <= max(1, res["greedy"] // 10), (name, res["greedy_mismatches"][:4])


def test_90_compare_prefix_and_multi_turn_with_reference():
    """The prefix-hit requests (A cold, B full hit, C system-prompt hit) give the same greedy tokens as on the
    reference server. Multi-turn: turn 1 is a free-form cold generation on both servers, and full-depth prefill is not
    bitwise reproducible (FULL_MODEL_VALIDATION §5.1): its first token was measured to be a near-tie (top-2 logprob
    margin 0.00-0.75 over four repeated cold prefills on one server, one of which flipped), so it is recorded, not
    asserted. Turn 2's prompt contains turn 1's tokens, so it is compared across servers only when turn 1 matched;
    within each server the live test already asserts hit == cold. Every request here runs alone, so packed prefill
    never applies (a pass needs two rows)."""
    a, b = _need_both("prefix_cache_hit")
    out = {}
    for k in "ABC":
        x, y = a["runs"][k]["result"]["token_ids"], b["runs"][k]["result"]["token_ids"]
        out[k] = dict(identical=x == y, divergence=first_divergence(x, y))
    a2, b2 = _need_both("multi_turn_hit")
    t1a, t1b = a2["runs"]["turn1"]["result"]["token_ids"], b2["runs"]["turn1"]["result"]["token_ids"]
    info = dict(turn1_identical=t1a == t1b, turn1_divergence=first_divergence(t1a, t1b))
    if t1a == t1b:
        for k in ("turn2_hit", "turn2_cold"):
            x, y = a2["runs"][k]["result"]["token_ids"], b2["runs"][k]["result"]["token_ids"]
            out[k] = dict(identical=x == y, divergence=first_divergence(x, y))
    save("compare_prefix_multi_turn", dict(summary=dict(out, info=info), reference=REF_DIR))
    assert all(v["identical"] for v in out.values()), out


def test_91_compare_long_and_mixed_with_reference():
    """The long prompt (alone in its prefill steps: never packed) must match exactly. The mixed PS-1 burst's greedy rows
    arrive naturally (timing decides the prefill steps): exact between two per-row launches, within the floor when
    either launch packs."""
    a, b = _need_both("long_prompt")
    out: Dict[str, Any] = dict(
        long_first=a["runs"]["first"]["result"]["token_ids"] == b["runs"]["first"]["result"]["token_ids"],
        long_again=a["runs"]["again"]["result"]["token_ids"] == b["runs"]["again"]["result"]["token_ids"],
    )
    a2, b2 = _need_both("mixed_ps1")
    ref = _ref_profile()
    exact = ref is not None and not FEAT["packed"] and not PROFILES[ref]["packed"] and _same_prefill(ref)
    cases = {c["i"]: c for c in _greedy_cases()}
    pairs = []
    for x, y in zip(a2["runs"]["greedy"], b2["runs"]["greedy"]):
        body = chat_body(cases[x["i"]]["text"], max_tokens=cases[x["i"]]["max_tokens"])
        pairs.append((f"mixed_greedy[{x['i']}]", body, x["token_ids"], y["token_ids"], False))
    res = _check_pairs("mixed_ps1", pairs, exact, False)
    out["mixed_greedy"] = res
    ss = [x["token_ids"] == y["token_ids"] for x, y in zip(a2["runs"]["sampled"], b2["runs"]["sampled"])]
    out["mixed_sampled_identical"] = f"{sum(ss)}/{len(ss)}"  # seeded: equal when the logits are
    save("compare_long_mixed", dict(summary=out, reference=REF_DIR, reference_profile=ref))
    assert out["long_first"] and out["long_again"], out
    _assert_pairs("mixed_ps1", res, exact, False)


GATED_WORKLOADS = (
    ("smoke_chats", ("a", "b")),
    ("greedy_speculation", ("a", "b")),
    ("p5_gated", ("short32", "short32_rep", "shared2k", "same6k", "mixed32", "mt16_t1", "mt16_t2")),
    ("t64_ps1_mix", ("mix",)),
    ("p5_pass_size_floor", tuple(str(b) for b in PASS_SIZES)),
    ("p5_logprob_replay", ("short32",)),
)


def test_92_compare_gated_with_reference():
    """Every gated workload's tokens against the reference launch's. Same prefill path (e.g. ``mtp_auto`` vs
    ``dsamp_pk``): greedy tokens identical (speculation, T64 and the device sampler are lossless), and seeded sampled
    tokens identical when both sample on the device. Packed vs per-row (``dsamp_pk`` vs ``dsamp``): every greedy
    difference a near-tie (probed on this live server)."""
    ref = _ref_profile()
    if ref is None:
        pytest.skip("needs MOTIF3_E2E_REFERENCE results")
    exact = _same_prefill(ref)
    sampled_comparable = exact and FEAT["dsamp"] == PROFILES[ref]["dsamp"]
    out: Dict[str, Any] = {}
    for name, keys in GATED_WORKLOADS:
        a, b = load(OUT_DIR, name), load(REF_DIR, name)
        if a is None or b is None or "runs" not in a or "runs" not in b:
            out[name] = "missing"
            continue
        for k in keys:
            if k not in a["runs"] or k not in b["runs"]:
                continue
            pairs = _pairs(a["runs"][k], b["runs"][k], f"{name}.{k}")
            out[f"{name}.{k}"] = _check_pairs(f"{name}.{k}", pairs, exact, sampled_comparable)
    save("compare_gated", dict(summary=out, reference=REF_DIR, reference_profile=ref, mode="exact" if exact else
                               "floor", sampled_comparable=sampled_comparable))  # fmt: skip
    assert any(isinstance(v, dict) for v in out.values()), out
    for k, v in out.items():
        if isinstance(v, dict):
            _assert_pairs(k, v, exact, sampled_comparable)


def test_93_compare_throughput_with_reference():
    """Decode throughput against the reference launch at every level: the speedup of the steady-window tok/s and the
    tokens (exact when both launches prefill alike: the "token-exact vs no-spec" check of an MTP launch). An MTP launch
    vs its non-speculating twin: c = 32 >= 1.4x on a T64 launch (G-serve), c = 1 / 8 >= 1.5x."""
    a, b = _need_both("decode_throughput")
    ref = _ref_profile()
    exact = _same_prefill(ref)
    out: Dict[str, Any] = {}
    for c in a["summary"]:
        if c not in b["summary"]:
            continue
        ta, tb = a["summary"][c]["steady"].get("tok_s"), b["summary"][c]["steady"].get("tok_s")
        pairs = _pairs(a["runs"][c], b["runs"][c], f"tput.c{c}") if "runs" in a and c in a["runs"] and c in b.get(
            "runs", {}) else []  # fmt: skip
        out[c] = dict(tok_s=[ta, tb], speedup=(ta / tb) if ta and tb else None,
                      tpot_p50_ms=[a["summary"][c]["tpot_ms"].get("p50"), b["summary"][c]["tpot_ms"].get("p50")],
                      acceptance=a["summary"][c].get("acceptance"), drafts=a["summary"][c].get("drafts"),
                      tokens=_check_pairs(f"tput.c{c}", pairs, exact, False) if pairs else None)  # fmt: skip
    save("compare_throughput", dict(summary=out, reference=REF_DIR, reference_profile=ref))
    for c, v in out.items():
        if v["tokens"] is not None:
            _assert_pairs(f"tput.c{c}", v["tokens"], exact, False)
    if SPEC and ref is not None and not PROFILES[ref]["spec"]:
        for c, v in out.items():
            if int(c) in (1, 8):
                assert v["speedup"] is not None and v["speedup"] >= BAR_LOW_C_SPEEDUP, (c, v)
            if int(c) == 32 and WIDE:
                assert v["speedup"] is not None and v["speedup"] >= BAR_C32_SPEEDUP, (c, v)


def test_94_compare_logprob_replay_with_reference():
    """Where the two launches' logprob replays diverge (a packed and a per-row launch), each launch's OWN
    log-probabilities of the two tokens at the first divergence: each picks its token by at most ``NEAR_TIE`` with the
    other launch's token among its top ``NEAR_TIE_RANK`` -- a near-tie seen from both sides. Identical replays when
    both launches prefill alike."""
    a, b = _need_both("p5_logprob_replay")
    ref = _ref_profile()
    exact = _same_prefill(ref)
    rows = []
    for i, (x, y) in enumerate(zip(a["runs"]["short32"]["results"], b["runs"]["short32"]["results"])):
        d = first_divergence(x["token_ids"], y["token_ids"])
        if d is None:
            continue
        row: Dict[str, Any] = dict(i=i, divergence=d)
        if d < min(len(x["token_ids"]), len(y["token_ids"])):
            ta, tb = int(x["token_ids"][d]), int(y["token_ids"][d])
            la = {int(t): float(v) for t, v in x["top_logprobs"][d]}
            lb = {int(t): float(v) for t, v in y["top_logprobs"][d]}
            row.update(token_this=ta, token_ref=tb, rank_of_ref_token_here=_rank(la, tb),
                       rank_of_this_token_in_ref=_rank(lb, ta),
                       margin_this=round(la[ta] - la[tb], 4) if ta in la and tb in la else None,
                       margin_ref=round(lb[tb] - lb[ta], 4) if ta in lb and tb in lb else None)  # fmt: skip
        rows.append(row)

    def ok(r):
        m1, m2 = r.get("margin_this"), r.get("margin_ref")
        r1, r2 = r.get("rank_of_ref_token_here"), r.get("rank_of_this_token_in_ref")
        return (m1 is not None and m2 is not None and 0 <= m1 <= NEAR_TIE and 0 <= m2 <= NEAR_TIE
                and r1 is not None and r1 <= NEAR_TIE_RANK and r2 is not None and r2 <= NEAR_TIE_RANK)  # fmt: skip

    margins = sorted(max(r["margin_this"], r["margin_ref"]) for r in rows if r.get("margin_this") is not None
                     and r.get("margin_ref") is not None)  # fmt: skip
    summary = dict(reference_profile=ref, mode="exact" if exact else "floor", identical=f"{32 - len(rows)}/32",
                   divergences=rows, worst_margin=margins[-1] if margins else None,
                   not_near_ties=[r["i"] for r in rows if not ok(r)])  # fmt: skip
    save("compare_logprob_replay", dict(summary=summary, reference=REF_DIR))
    if exact:
        assert not rows, summary
    else:
        assert not summary["not_near_ties"], summary


# ======================================================================================================================
# 99. the server's counters at shutdown (run after SIGTERM; no server needed)
# ======================================================================================================================
def _last_dict(text: str, pattern: str) -> Optional[Dict[str, Any]]:
    found = re.findall(pattern, text)
    if not found:
        return None
    raw = found[-1]
    try:
        return json.loads(raw)
    except ValueError:
        return ast.literal_eval(raw)


def test_99_shutdown_counters():
    """After the server's SIGTERM shutdown: the generator's / bridge's counters (packed passes and no solo fallback on a
    packed launch, T64 steps on a T64 launch, no non-greedy verify row, the PS-1 canary 0), no traceback before the
    shutdown, and the devices closed."""
    text = server_log_text()
    if text is None:
        pytest.skip("needs MOTIF3_E2E_SERVER_LOG")
    gen = _last_dict(text, r"Motif-3 generator: (\{[^}]*\})")
    if gen is None:
        pytest.skip("the server has not shut down yet (no 'Motif-3 generator:' counters)")
    spec = _last_dict(text, r"Motif-3 speculation: (\{[^}]*\})")
    samp = _last_dict(text, r"Motif-3 device sampling: (\{[^}]*\})")
    cut = text.find("trigger received signal=SIGTERM")
    before = text if cut < 0 else text[:cut]
    tail = re.findall(r"Motif-3 speculation: \{[^}]*\} (.*)", text)
    summary = dict(
        generator=gen,
        speculation=spec,
        speculation_tail=tail[-1] if tail else None,
        device_sampling=samp,
        tracebacks_before_shutdown=before.count("Traceback (most recent call last)"),
        errors_before_shutdown=[ln[:300] for ln in before.splitlines() if " ERROR " in ln][:10],
        ps1_canary_lines=len(re.findall(r"not speculable", text)),
        devices_closed="Closing devices in cluster completed" in text,
        sigterm=cut >= 0,
    )
    save("shutdown_counters", dict(summary=summary))
    assert summary["sigterm"] and summary["devices_closed"], summary
    assert summary["tracebacks_before_shutdown"] == 0, summary["errors_before_shutdown"]
    if FEAT["packed"]:
        assert gen["packed_passes"] > 0 and gen["packed_pk1_passes"] > 0, gen
        assert gen["packed_solo_fallbacks"] == 0 and gen["packed_plan_errors"] == 0, gen
    else:
        assert gen.get("packed_passes", 0) == 0, gen
    if WIDE:
        assert gen["wide_steps"] > 0 and gen["wide_drafts"] > 0, gen
    if SPEC and FEAT["verify"] == "auto":
        assert gen["auto_t32_verifies"] > 0, gen  # verify steps below c* stay on T32
        assert gen["overflow_passes"] == 0, gen  # bridge traffic never overflows in "auto" (R-E6)
    if not SPEC:
        assert gen.get("verify_steps", 0) == 0 and gen.get("wide_steps", 0) == 0, gen
    if SPEC and samp is not None:
        assert samp.get("nongreedy_verify_rows", 0) == 0, samp
    if SPEC:
        assert summary["ps1_canary_lines"] == 0, summary
