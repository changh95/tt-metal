# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end validation of a RUNNING Motif-3 vLLM server with chunked prefill, prefix caching and MTP
self-speculative decoding (``docs/features/FEATURES_DESIGN.md`` §5.4; results in ``docs/FEATURES_RESULTS.md``).

This module is an HTTP client only: it never imports ttnn or opens a device. Run it with the devices hidden (the
tt-metal root conftest opens the UMD cluster even for collection, which would disturb the server that owns the chips)::

    MOTIF3_E2E_PROFILE=all MOTIF3_E2E_OUT=<dir> MOTIF3_E2E_SERVER_LOG=<vllm log> \\
    scripts/hostrun.sh -n e2e_all -- python -m pytest -p no:cacheprovider -s -q --timeout=0 \\
        models/demos/motif3/tests/test_vllm_features_e2e.py

Environment:

* ``MOTIF3_E2E_URL`` (default ``http://127.0.0.1:8000``): the server. Live tests skip when it does not answer.
* ``MOTIF3_E2E_PROFILE``: what the server under test enables (the launch of ``logs/serve/features/features_hold.sh``):
  ``all`` (chunked prefill 8128/8128 + prefix caching + MTP K=1, the production launch), ``nomtp`` (no
  ``--speculative-config``), ``off`` (``MOTIF3_*=0`` with the draft-1 TIS flags). The assertions follow it.
* ``MOTIF3_E2E_OUT``: directory for one JSON per live test (raw results and summaries).
* ``MOTIF3_E2E_REFERENCE``: the ``MOTIF3_E2E_OUT`` of a run on another server (normally ``off``). The ``test_9*``
  comparison tests read both directories and need no server.
* ``MOTIF3_E2E_SERVER_LOG``: the server's log. Checked for the bridge's feature line and for the plugin's PS-1
  canary (a verify step that held a sampled request logs "not speculable").
* ``MOTIF3_E2E_SMOKE_REFERENCE`` (default ``logs/serve/results/final/smoke.json``): the draft-1 TIS server's chat
  outputs (SERVING_SMOKE §3; 8/8 greedy texts character-identical to FULL_MODEL_VALIDATION §3).

Live tests (in order; each records ``<name>.json``):

1. ``test_server_config``: ``/v1/models``; the bridge's ``Motif-3 features:`` line matches the profile.
2. ``test_smoke_chats``: the SERVING_SMOKE §3 workload (4 prompts x greedy / T=1.0 / T=0.6 seeded x thinking off /
   on; 12 concurrent each): answers judged correct, greedy texts identical to the draft-1 reference, sampled texts
   compared to it (seeded) and never collapsed onto the greedy text.
3. ``test_greedy_speculation_lossless``: the 8 greedy chats alone (4 concurrent, thinking off then on), so every row
   speculates on ``all``: texts identical to the draft-1 reference; drafts and acceptance from ``/metrics``.
4. ``test_prefix_cache_hit``: a ~6K-token system prompt + question twice (second TTFT far lower, identical text,
   ``vllm:prefix_cache_hits``), then a different question behind the same system prompt.
5. ``test_multi_turn_decode_written_hit``: turn 2 = turn 1's prompt + its 192 DECODE-generated tokens + a new turn
   (token ids through ``/v1/completions``): the hit covers decode-written blocks and must equal the same prompt sent
   cold (``cache_salt``). Decode wrote those blocks from one lane's DP row only (plus every row under KV-R); the
   replicated prefill reads them on all 4 rows and the MoE reduce-scatter mixes the rows, so without KV-R any hit on
   them is wrong whichever lane hits (design §3.4, review R7). A filler request admitted between the turns also moves
   turn 2 to another persistent-batch row (by the plugin's slot rule, another state slot and DP row; the client cannot
   observe lanes).
6. ``test_long_prompt_chunked``: a ~30K-token needle prompt while another request decodes: the needle is answered,
   TTFT, the other request's worst inter-token gap (chunked prefill bounds the decode stall); then the same prompt
   again (a full prefix hit).
7. ``test_mixed_concurrency_ps1``: 32 concurrent: 16 sampled (submitted first, ignore_eos, 320 tokens, seeded) + 16
   greedy needle / arithmetic requests. PS-1: while a sampled request is live no draft is scheduled
   (``vllm:spec_decode_num_drafts`` delta 0 on ``all``); the greedy answers are correct; then the same 16 greedy
   requests alone speculate (drafts > 0) and give identical texts (with prefix hits); the server log has no
   "not speculable" warning.
   ``test_spec_refuses_logprobs``: a ``logprobs`` request is refused (HTTP 400) on a speculating launch, served
   otherwise.
8. ``test_burst_ttft``: 32 short unique prompts at once, then 32 sharing a ~2K-token system prompt: TTFT stats.
9. ``test_decode_throughput``: greedy, thinking on, 256 tokens (ignore_eos), c = 1 / 8 / 32: per-user TPOT, the
   steady-window aggregate tokens/s, the acceptance (``all``).

Comparison tests (``MOTIF3_E2E_OUT`` vs ``MOTIF3_E2E_REFERENCE``, no server): greedy outputs of the same requests on
the two servers (prefix hit, multi-turn, long prompt, mixed greedy rows) are identical.
"""

from __future__ import annotations

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
from typing import Any, Dict, List, Optional, Sequence

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
PROFILES = {
    "all": dict(chunked=True, prefix=True, spec=True),
    "nomtp": dict(chunked=True, prefix=True, spec=False),
    "off": dict(chunked=False, prefix=False, spec=False),
}
if PROFILE not in PROFILES:
    raise ValueError(f"MOTIF3_E2E_PROFILE must be one of {sorted(PROFILES)}, got {PROFILE!r}")
FEAT = PROFILES[PROFILE]
BLOCK = 64
BUDGET = 8128  # --max-num-batched-tokens = --long-prefill-token-threshold of the feature profiles

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


async def stream(session, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Streaming POST; returns status, text, token_ids, usage, finish_reason, chunk_times (s after send), ttft_s,
    latency_s, t_send / t_first / t_last (perf_counter), error."""
    body = dict(body, stream=True)
    body.setdefault("stream_options", {"include_usage": True})
    res: Dict[str, Any] = dict(status=None, text="", token_ids=[], usage=None, finish_reason=None, chunk_times=[],
                               chunk_ntok=[], error=None)  # fmt: skip
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
)


def mdelta(m1: Dict[str, float], m0: Dict[str, float]) -> Dict[str, float]:
    d = {k.replace("vllm:", "").replace("_total", ""): m1.get(k, 0.0) - m0.get(k, 0.0) for k in METRIC_KEYS}
    if d["spec_decode_num_draft_tokens"] > 0:
        d["acceptance"] = d["spec_decode_num_accepted_tokens"] / d["spec_decode_num_draft_tokens"]
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
# 1. server configuration
# ======================================================================================================================
def test_server_config(server):
    async def go(s):
        return await get_json(s, "/v1/models"), await metrics(s)

    models, m = run(_session_do(go))
    ids = [x["id"] for x in models["data"]]
    assert MODEL in ids, ids
    assert [x.get("max_model_len") for x in models["data"]] == [32768]
    summary = dict(models=ids, max_model_len=32768, metrics_present=sorted(k for k in m if "spec_decode" in k)[:6])
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
        assert re.search(rf"\bspec_tokens={1 if FEAT['spec'] else 0}\b", line), line
        mode = ("all" if FEAT["prefix"] else "row") + ("_split" if FEAT["spec"] else "")
        assert re.search(rf"\bkv_write={mode}\b", line), (mode, line)
        warm = re.findall(r"warmup prefill (sp[01]) bucket (\d+)", text)
        summary["warmup_prefill_shapes"] = len(warm)
        took = re.findall(r"init engine \(profile, create kv cache, warmup model\) took ([0-9.]+)", text)
        summary["init_engine_s"] = float(took[-1]) if took else None
        caps = re.findall(r"Motif-3 KV pool: (.*)", text)
        summary["kv_pool"] = caps[-1] if caps else None
        if FEAT["spec"]:
            assert summary["kv_pool"] and "MTP" in summary["kv_pool"], summary["kv_pool"]
    save("server_config", dict(summary=summary))


# ======================================================================================================================
# 2. the SERVING_SMOKE §3 chat workload, compared with the draft-1 server
# ======================================================================================================================
def _smoke_reference() -> Optional[Dict[tuple, Dict[str, Any]]]:
    if not SMOKE_REF.is_file():
        return None
    d = json.loads(SMOKE_REF.read_text())
    return {(c["prompt"], c["mode"], c["thinking"]): c for c in d["chat"]}


async def _one_chat(s, pname: str, mode: str, thinking: Optional[bool], max_tokens: int) -> Dict[str, Any]:
    r = await stream(s, "/v1/chat/completions", chat_body(SMOKE_PROMPTS[pname], max_tokens=max_tokens,
                                                          thinking=thinking, **SMOKE_MODES[mode]))  # fmt: skip
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


def test_smoke_chats(server):
    async def go(s):
        m0 = await metrics(s)
        t0 = time.perf_counter()
        a = await asyncio.gather(*[_one_chat(s, p, m, False, 512) for m in SMOKE_MODES for p in SMOKE_PROMPTS])
        w0 = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = await asyncio.gather(*[_one_chat(s, p, m, None, 3072) for m in SMOKE_MODES for p in SMOKE_PROMPTS])
        w1 = time.perf_counter() - t0
        return list(a) + list(b), w0, w1, mdelta(await metrics(s), m0)

    chats, w0, w1, md = run(_session_do(go))
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
        wall_s=[round(w0, 1), round(w1, 1)],
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
    save("smoke_chats", dict(summary=summary, chats=chats))
    assert summary["status_ok"] == 24, [c["error"] for c in chats if c["error"]]
    assert all(c["judge"].get("ok") for c in greedy), [
        (c["prompt"], c["thinking"]) for c in greedy if not c["judge"]["ok"]
    ]
    assert summary["stopped_on_eos"] == 24
    assert len(collapsed) <= 2, f"sampled outputs equal to the greedy text (argmax collapse?): {collapsed}"
    if ref is not None:
        assert len(g_cmp) == len(greedy), f"greedy texts differ from the draft-1 server: {cmp['different']}"


# ======================================================================================================================
# 3. greedy-only chats: every row speculates on the 'all' server
# ======================================================================================================================
def test_greedy_speculation_lossless(server):
    async def go(s):
        m0 = await metrics(s)
        a = await asyncio.gather(*[_one_chat(s, p, "greedy", False, 512) for p in SMOKE_PROMPTS])
        m1 = await metrics(s)
        b = await asyncio.gather(*[_one_chat(s, p, "greedy", None, 3072) for p in SMOKE_PROMPTS])
        m2 = await metrics(s)
        return list(a) + list(b), mdelta(m1, m0), mdelta(m2, m1)

    chats, md_off, md_on = run(_session_do(go))
    ref = _smoke_reference()
    cmp = _compare_smoke(chats, ref)
    summary = dict(
        identical_to_draft1=f"{len(cmp['identical'])}/{len(chats)}" if ref else None,
        different=cmp["different"],
        judged_ok=sum(bool(c["judge"].get("ok")) for c in chats),
        tpot_ms=[round(1e3 * c["tpot_s"], 1) if c["tpot_s"] else None for c in chats],
        tokens=[c["completion_tokens"] for c in chats],
        metrics_thinking_off=md_off,
        metrics_thinking_on=md_on,
    )
    save("greedy_speculation", dict(summary=summary, chats=chats))
    assert all(c["status"] == 200 and not c["error"] for c in chats)
    assert all(c["judge"].get("ok") for c in chats)
    if FEAT["spec"]:
        assert md_off["spec_decode_num_drafts"] > 0 and md_on["spec_decode_num_drafts"] > 0, (md_off, md_on)
        assert md_on.get("acceptance", 0) > 0.5, md_on
    else:
        assert md_off["spec_decode_num_drafts"] == 0 and md_on["spec_decode_num_drafts"] == 0
    if ref is not None:
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
        for name, q in (("A", q1), ("B", q1), ("C", q2)):
            m0 = await metrics(s)
            r = await stream(s, "/v1/chat/completions", chat_body(q, system=system, max_tokens=48))
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
        m0 = await metrics(s)
        t1 = await stream(s, "/v1/completions", comp(p1, max_tokens=192, ignore_eos=True))
        out["turn1"] = dict(result=slim(t1), metrics=mdelta(await metrics(s), m0))
        g1 = list(t1["token_ids"])
        p2 = p1 + g1 + turn2
        # A filler request is admitted first and takes persistent-batch row 0 (where turn 1 ran), so turn 2 takes the
        # next row; vllm-tt-plugin _alloc_prefill_state_slots maps a row to the same state slot when it is free, and
        # the bridge's LaneMap deals slots round-robin over the DP rows (slot 0 -> lane 0, slot 1 -> lane 8).
        gen0 = (await metrics(s)).get("vllm:generation_tokens_total", 0.0)
        filler_task = asyncio.ensure_future(
            stream(s, "/v1/completions", comp("Count upwards: 1, 2, 3,", max_tokens=320, ignore_eos=True))
        )
        deadline = time.perf_counter() + 120
        while not filler_task.done() and time.perf_counter() < deadline:  # until the filler is decoding
            await asyncio.sleep(0.25)
            if (await metrics(s)).get("vllm:generation_tokens_total", 0.0) >= gen0 + 3:
                break
        m1 = await metrics(s)
        hit = await stream(s, "/v1/completions", comp(p2, max_tokens=64))
        out["turn2_hit"] = dict(result=slim(hit), metrics=mdelta(await metrics(s), m1))
        m2 = await metrics(s)
        cold = await stream(s, "/v1/completions", comp(p2, max_tokens=64, cache_salt=f"cold-{uuid.uuid4().hex}"))
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
        ticker = asyncio.ensure_future(
            stream(s, "/v1/completions", dict(model=MODEL, prompt="Count upwards: 1, 2, 3,", max_tokens=800,
                                              temperature=0.0, ignore_eos=True, return_token_ids=True))
        )  # fmt: skip
        await asyncio.sleep(3.0)  # the ticker is decoding
        m0 = await metrics(s)
        t_long = time.perf_counter()
        r1 = await stream(s, "/v1/chat/completions", chat_body(text, max_tokens=48))
        out["first"] = dict(result=slim(r1), metrics=mdelta(await metrics(s), m0), t_send=r1["t_send"],
                            t_first=r1["t_first"])  # fmt: skip
        m1 = await metrics(s)
        r2 = await stream(s, "/v1/chat/completions", chat_body(text, max_tokens=48))
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


async def _greedy_case(s, c, delay=0.0):
    await asyncio.sleep(delay)
    r = await stream(s, "/v1/chat/completions", chat_body(c["text"], max_tokens=c["max_tokens"]))
    out = slim(r)
    ans = answer_of(r)
    out.update(
        i=c["i"],
        kind=c["kind"],
        expected=c["answer"],
        t_first=r["t_first"],
        t_last=r["t_last"],
        correct=(c["answer"] in ans.replace(",", "")) if c["kind"] == "arith" else (c["answer"] in ans),
    )
    return out


async def _sampled_case(s, c, delay=0.0):
    await asyncio.sleep(delay)
    r = await stream(s, "/v1/chat/completions",
                     chat_body(c["text"], max_tokens=c["max_tokens"], ignore_eos=True, **c["samp"]))  # fmt: skip
    out = slim(r)
    out.update(i=c["i"], samp=c["samp"], t_first=r["t_first"], t_last=r["t_last"], t_send=r["t_send"])
    return out


def test_mixed_concurrency_ps1(server):
    g_cases, s_cases = _greedy_cases(), _sampled_cases()

    async def go(s):
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
            stream(s, "/v1/chat/completions", chat_body(c["text"], max_tokens=c["max_tokens"], ignore_eos=True))
            for c in s_cases[:4]
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
    assert all(x["completion_tokens"] == 320 for x in sampled)
    assert sum(collapsed) == 0, "a sampled request produced exactly its greedy text (argmax collapse)"
    assert all(same_b), [x["i"] for x, ok in zip(greedy, same_b) if not ok]
    if FEAT["spec"]:
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
        async with s.post(BASE + "/v1/chat/completions", json=body) as r:
            return r.status, (await r.text())[:2000]

    status, text = run(_session_do(go))
    save("spec_refusals", dict(summary=dict(status=status, body=text[:400])))
    if FEAT["spec"]:
        assert status == 400 and "logprobs" in text and "peculative" in text, (status, text)
    else:
        assert status == 200 and '"top_logprobs"' in text, (status, text)


# ======================================================================================================================
# 8. TTFT under a burst
# ======================================================================================================================
def test_burst_ttft(server):
    shared = _handbook(2000, seed=123)

    async def go(s):
        out = {}
        m0 = await metrics(s)
        t0 = time.perf_counter()
        a = await asyncio.gather(*[stream(s, "/v1/chat/completions",
                                          chat_body(f"[{uuid.uuid4().hex[:8]}] {QUESTIONS[i]}", max_tokens=8))
                                   for i in range(32)])  # fmt: skip
        out["unique"] = dict(
            results=[slim(r) for r in a], wall_s=time.perf_counter() - t0, metrics=mdelta(await metrics(s), m0)
        )
        m1 = await metrics(s)
        t0 = time.perf_counter()
        salt = uuid.uuid4().hex[:8]
        b = await asyncio.gather(*[stream(s, "/v1/chat/completions",
                                          chat_body(f"{QUESTIONS[i]} (ref {salt})", system=shared, max_tokens=8))
                                   for i in range(32)])  # fmt: skip
        out["shared_2k"] = dict(
            results=[slim(r) for r in b], wall_s=time.perf_counter() - t0, metrics=mdelta(await metrics(s), m1)
        )
        return out

    res = run(_session_do(go))
    summary = {
        k: dict(ttft=stats([r["ttft_s"] for r in v["results"]]), wall_s=round(v["wall_s"], 2),
                prompt_tokens=stats([r["prompt_tokens"] for r in v["results"]]),
                prefix_hits=v["metrics"]["prefix_cache_hits"], status_ok=sum(r["status"] == 200 for r in v["results"]))
        for k, v in res.items()
    }  # fmt: skip
    save("burst_ttft", dict(summary=summary, runs=res))
    for v in res.values():
        assert all(r["status"] == 200 and not r["error"] for r in v["results"])
    if FEAT["prefix"]:
        assert res["shared_2k"]["metrics"]["prefix_cache_hits"] > 20 * 1900, summary


# ======================================================================================================================
# 9. decode throughput at c = 1 / 8 / 32
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


async def _tput_case(s, q: str, max_tokens: int):
    return await stream(s, "/v1/chat/completions",
                        chat_body(q, max_tokens=max_tokens, thinking=None, ignore_eos=True,
                                  cache_salt=uuid.uuid4().hex))  # fmt: skip


def test_decode_throughput(server):
    levels = [int(x) for x in os.environ.get("MOTIF3_E2E_TPUT_LEVELS", "1,8,32").split(",")]
    max_tokens = int(os.environ.get("MOTIF3_E2E_TPUT_TOKENS", "256"))

    async def go(s):
        out = {}
        for c in levels:
            m0 = await metrics(s)
            load0 = os.getloadavg()  # host CPU contention slows the server's host side (sampling, input build)
            t0 = time.perf_counter()
            rs = await asyncio.gather(*[_tput_case(s, QUESTIONS[i % len(QUESTIONS)], max_tokens) for i in range(c)])
            wall = time.perf_counter() - t0
            md = mdelta(await metrics(s), m0)
            out[str(c)] = dict(raw=list(rs), wall_s=wall, metrics=md, loadavg=[load0, os.getloadavg()])
        return out

    res = run(_session_do(go))
    summary = {}
    for c, v in res.items():
        rs = v["raw"]
        tp = [tpot(r) for r in rs]
        st = _steady(rs)
        summary[c] = dict(
            status_ok=sum(r["status"] == 200 and not r["error"] for r in rs),
            tokens=sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in rs),
            tpot_ms=stats([1e3 * t for t in tp if t]),
            per_user_tok_s=stats([1.0 / t for t in tp if t]),
            steady=st,
            wall_s=round(v["wall_s"], 2),
            ttft=stats([r["ttft_s"] for r in rs]),
            acceptance=v["metrics"].get("acceptance"),
            drafts=v["metrics"]["spec_decode_num_drafts"],
            loadavg_1m=[round(x[0], 1) for x in v["loadavg"]],
            multi_token_chunks=sum(sum(1 for k in r["chunk_ntok"] if k > 1) for r in rs),
        )
    save(
        "decode_throughput",
        dict(
            summary=summary,
            runs={c: dict(results=[slim(r) for r in v["raw"]], metrics=v["metrics"]) for c, v in res.items()},
        ),
    )
    for c, v in summary.items():
        assert v["status_ok"] == int(c)
        assert v["tokens"] == int(c) * max_tokens
        if FEAT["spec"] and int(c) <= 16:  # at 32 live lanes there is no idle lane to pack a draft into (by design)
            assert v["drafts"] > 0 and (v["acceptance"] or 0) > 0.4, (c, v)
        if not FEAT["spec"]:
            assert v["drafts"] == 0 and v["multi_token_chunks"] == 0, (c, v)


# ======================================================================================================================
# comparisons with the reference server's results (no server needed)
# ======================================================================================================================
def _need_both(name: str):
    a, b = load(OUT_DIR, name), load(REF_DIR, name)
    if a is None or b is None:
        pytest.skip(f"{name}: needs MOTIF3_E2E_OUT and MOTIF3_E2E_REFERENCE results")
    return a, b


def test_90_compare_prefix_and_multi_turn_with_reference():
    """The prefix-hit requests (A cold, B full hit, C system-prompt hit) give the same greedy tokens as on the
    reference server. Multi-turn: turn 1 is a free-form cold generation on both servers, and full-depth prefill is not
    bitwise reproducible (FULL_MODEL_VALIDATION §5.1): its first token was measured to be a near-tie (top-2 logprob
    margin 0.00-0.75 over four repeated cold prefills on one server, one of which flipped), so it is recorded, not
    asserted. Turn 2's prompt contains turn 1's tokens, so it is compared across servers only when turn 1 matched;
    within each server the live test already asserts hit == cold."""
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
    a, b = _need_both("long_prompt")
    out = dict(
        long_first=a["runs"]["first"]["result"]["token_ids"] == b["runs"]["first"]["result"]["token_ids"],
        long_again=a["runs"]["again"]["result"]["token_ids"] == b["runs"]["again"]["result"]["token_ids"],
    )
    a2, b2 = _need_both("mixed_ps1")
    same = [x["token_ids"] == y["token_ids"] for x, y in zip(a2["runs"]["greedy"], b2["runs"]["greedy"])]
    out["mixed_greedy_identical"] = f"{sum(same)}/{len(same)}"
    out["mixed_greedy_different"] = [x["i"] for x, ok in zip(a2["runs"]["greedy"], same) if not ok]
    ss = [x["token_ids"] == y["token_ids"] for x, y in zip(a2["runs"]["sampled"], b2["runs"]["sampled"])]
    out["mixed_sampled_identical"] = f"{sum(ss)}/{len(ss)}"  # seeded: equal when the logits are
    save("compare_long_mixed", dict(summary=out, reference=REF_DIR))
    assert out["long_first"] and out["long_again"] and all(same), out
