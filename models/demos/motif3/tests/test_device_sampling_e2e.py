# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end validation of exact DEVICE SAMPLING through a RUNNING Motif-3 vLLM server (``docs/sampling/
DEVICE_SAMPLER.md`` §11): the production default launch (lead decision 1: chunked prefill + prefix caching +
``"sample_on_device_mode": "decode_only"``, no speculation) and the opt-in MTP launch (the same +
``--speculative-config``, sampled rows decoding without speculation under PS-1, sampled on device as well).

HTTP client only: it never imports ttnn or opens a device. Run it with the devices hidden (the tt-metal root conftest
would open the chips the server owns)::

    MOTIF3_DS_PROFILE=dsamp MOTIF3_DS_OUT=<dir> MOTIF3_DS_SERVER_LOG=<vllm log> \\
    scripts/hostrun.sh -t 7000 -n ds_e2e -- python -m pytest -p no:cacheprovider -s -q --timeout=0 -o addopts="" \\
        models/demos/motif3/tests/test_device_sampling_e2e.py

Servers: ``logs/serve/sampling/start_sampling_server.sh dsamp|dsamp_mtp`` (port 8013, the features launch pattern).

Environment: ``MOTIF3_DS_URL`` (default ``http://127.0.0.1:8013``), ``MOTIF3_DS_PROFILE`` (``dsamp`` | ``dsamp_mtp``),
``MOTIF3_DS_OUT`` (one JSON per test), ``MOTIF3_DS_SERVER_LOG`` (the bridge's ``Motif-3 device sampling:`` counter
lines, logged every ``MOTIF3_SAMPLING_LOG_EVERY`` device steps), ``MOTIF3_DS_BUDGET`` (the expected chunk budget =
threshold, e.g. 8064: the generator's feature line must show it and the bridge must log no serving-config warning),
``MOTIF3_DS_SMOKE_REFERENCE`` (the draft-1 TIS server's chats, ``logs/serve/results/final/smoke.json``: its greedy
texts must be reproduced -- device greedy is ``torch.argmax`` like the host's), ``MOTIF3_DS_REFERENCE`` (another
profile's ``MOTIF3_DS_OUT``, for the cross-profile greedy comparison), ``MOTIF3_DS_TPUT_LEVELS`` (``1,8,32``).

Tests (in order; each records ``<name>.json``):

1. ``test_00_server_config``: ``/v1/models``; the server log shows the plugin running ``sample_on_device_mode=
   decode_only`` and the bridge's device sampler on; the bridge's feature line matches the profile (and the budget
   ``MOTIF3_DS_BUDGET``, without a serving-config warning).
2. ``test_10_chats``: the SERVING_SMOKE §3 prompts x {greedy, T 1.0 / top-p 0.95, T 1.0 / top_p 1 (the full-vocab
   Gumbel path), T 0.6 / top-p 0.95, T 0.8 / top-k 20} x thinking {off, on}: judged answers (scattering, 서울, $505,
   the palindrome code runs) for greedy and for most sampled chats, no degenerate repetition, greedy texts identical to
   the draft-1 host-sampled server, sampled texts never the greedy text.
3. ``test_20_seeded_reproducible``: 12 seeded requests (top-p, top_p = 1, top-k, T 0.6), run concurrently, then one by
   one in reverse order, then mixed with 20 unseeded requests: identical token ids (the device draw is a pure function
   of (seed, position); every run prefills cold through a fresh ``cache_salt``).
4. ``test_30_mixed_batch``: 32 concurrent: greedy, top-p 0.95, top_p = 1, top-k rows, and (``dsamp``) ``logprobs=0``
   rows; twice: greedy rows equal their solo outputs, seeded rows identical between the two batches, the raw logprobs
   finite and <= 0.
5. ``test_40_host_routed``: a penalized request (presence_penalty: host-only) next to seeded sampled ones: all served;
   the bridge counted host-sampled steps.
6. ``test_50_counters``: the bridge / generator / sampler counters of the server's latest log line: device-sampled
   steps, Gumbel lanes (never a host fallback), the host-fallback rate (expected ~1 % of steps at T 1.0 / top-p 0.95),
   no non-greedy verify row (MTP).
7. ``test_60_tpot``: decode TPOT at c = 1 / 8 / 32 (256 tokens, ``ignore_eos``, thinking on) for greedy, T 1.0 / top-p
   0.95 and top_p = 1 traffic: sampled TPOT ~ greedy (host sampling cost ~170 ms per step at 32 rows, 264-271 ms TPOT).
8. ``test_70_greedy_speculation`` (``dsamp_mtp``): greedy-only traffic still speculates (drafts, acceptance > 0.4)
   with device sampling on: verify steps carry ``sampling_params`` and keep the argmax path; texts equal the ``dsamp``
   greedy texts (``MOTIF3_DS_REFERENCE``).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

PROJECT = Path(__file__).resolve().parents[5]
BASE = os.environ.get("MOTIF3_DS_URL", "http://127.0.0.1:8013").rstrip("/")
PROFILE = os.environ.get("MOTIF3_DS_PROFILE", "dsamp").strip().lower()
if PROFILE not in ("dsamp", "dsamp_mtp"):
    raise ValueError(f"MOTIF3_DS_PROFILE must be dsamp or dsamp_mtp, got {PROFILE!r}")
SPEC = PROFILE == "dsamp_mtp"
OUT_DIR = os.environ.get("MOTIF3_DS_OUT")
REF_DIR = os.environ.get("MOTIF3_DS_REFERENCE")
SERVER_LOG = os.environ.get("MOTIF3_DS_SERVER_LOG")
SMOKE_REF = Path(os.environ.get("MOTIF3_DS_SMOKE_REFERENCE",
                                str(PROJECT / "logs" / "serve" / "results" / "final" / "smoke.json")))  # fmt: skip
MODEL = "Motif-Technologies/Motif-3"
SYSTEM = "You are a helpful assistant."
EOS_IDS = {0, 3, 6}
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
MODES = {
    "greedy": dict(temperature=0.0, top_p=1.0),
    "t1.0_p0.95": dict(temperature=1.0, top_p=0.95, seed=1234),
    "t1.0_p1": dict(temperature=1.0, top_p=1.0, seed=77),  # the full-vocab Gumbel path (lead decision 4)
    "t0.6_p0.95": dict(temperature=0.6, top_p=0.95, seed=1234),
    "t0.8_k20": dict(temperature=0.8, top_p=1.0, top_k=20, seed=5),
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


def log(msg: str) -> None:
    print(f"[ds-e2e {PROFILE} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ======================================================================================================================
# HTTP client (aiohttp, SSE streaming with per-chunk arrival times)
# ======================================================================================================================
def _timeout():
    import aiohttp

    return aiohttp.ClientTimeout(total=7200, sock_read=7200)


def chat_body(user: str, *, max_tokens: int = 512, thinking: Optional[bool] = False, **extra) -> Dict[str, Any]:
    """A /v1/chat/completions body (``thinking`` None = the server default, on); ``extra``: sampling fields."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    b: Dict[str, Any] = dict(model=MODEL, messages=msgs, max_tokens=max_tokens)
    b.setdefault("temperature", 0.0)
    if thinking is not None:
        b["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    b.update(return_token_ids=True, skip_special_tokens=False)
    b.update(extra)
    return b


async def stream(session, body: Dict[str, Any]) -> Dict[str, Any]:
    """Streaming chat POST: status, text, token_ids, logprobs (per token, when asked), usage, finish_reason,
    chunk_times (s after send), ttft_s, error."""
    body = dict(body, stream=True)
    body.setdefault("stream_options", {"include_usage": True})
    res: Dict[str, Any] = dict(status=None, text="", token_ids=[], logprobs=[], usage=None, finish_reason=None,
                               chunk_times=[], chunk_ntok=[], error=None)  # fmt: skip
    t0 = time.perf_counter()
    res["t_send"] = t0
    try:
        async with session.post(BASE + "/v1/chat/completions", json=body) as r:
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
                            d = ch.get("delta") or {}
                            piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                            tids = ch.get("token_ids") or []
                            if piece or tids:
                                res["chunk_times"].append(now)
                                res["chunk_ntok"].append(len(tids))
                                res["text"] += piece
                                res["token_ids"].extend(int(t) for t in tids)
                            lp = (ch.get("logprobs") or {}).get("content") or []
                            res["logprobs"].extend(float(e["logprob"]) for e in lp)
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


async def metrics(session) -> Dict[str, float]:
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


def mdelta(m1: Dict[str, float], m0: Dict[str, float]) -> Dict[str, float]:
    keys = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total", "vllm:prefix_cache_hits_total",
            "vllm:generation_tokens_total", "vllm:request_success_total")  # fmt: skip
    d = {k.replace("vllm:", "").replace("_total", ""): m1.get(k, 0.0) - m0.get(k, 0.0) for k in keys}
    if d["spec_decode_num_draft_tokens"] > 0:
        d["acceptance"] = d["spec_decode_num_accepted_tokens"] / d["spec_decode_num_draft_tokens"]
    return d


def tpot(r: Dict[str, Any]) -> Optional[float]:
    n = (r.get("usage") or {}).get("completion_tokens")
    ct = r.get("chunk_times") or []
    if not n or n < 2 or len(ct) < 2:
        return None
    return (ct[-1] - ct[0]) / (n - 1)


def stats(vals: Sequence[Optional[float]]) -> Dict[str, Any]:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {"n": 0}

    def q(p):
        k = (len(v) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)

    return dict(n=len(v), mean=sum(v) / len(v), p50=q(0.5), p90=q(0.9), min=v[0], max=v[-1])


def answer_of(r: Dict[str, Any]) -> str:
    text = r.get("text") or ""
    text = text.split("</think>", 1)[1] if "</think>" in text else text
    for stop in ("<|endofturn|>", "<|endoftext|>", "<|user|>"):
        text = text.split(stop, 1)[0]
    return text.strip()


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


def degenerate(ids: Sequence[int], n: int = 8, limit: int = 12) -> bool:
    """A decoding loop: some n-gram of token ids repeats more than ``limit`` times."""
    seen: Dict[tuple, int] = {}
    for i in range(len(ids) - n + 1):
        k = tuple(ids[i : i + n])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] > limit:
            return True
    return False


def slim(r: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in r.items() if k not in ("chunk_times", "chunk_ntok", "t_send", "t_first", "t_last")}
    out["tpot_s"] = tpot(r)
    out["completion_tokens"] = (r.get("usage") or {}).get("completion_tokens")
    return out


def save(name: str, payload: Dict[str, Any]) -> None:
    payload = dict(payload, profile=PROFILE, url=BASE, saved=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if OUT_DIR:
        d = Path(OUT_DIR)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    log(f"{name}: {json.dumps(payload.get('summary', {}), ensure_ascii=False)[:2500]}")


def load(dirname: Optional[str], name: str) -> Optional[Dict[str, Any]]:
    if not dirname:
        return None
    p = Path(dirname) / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def server_log_text() -> Optional[str]:
    if not SERVER_LOG or not Path(SERVER_LOG).is_file():
        return None
    return Path(SERVER_LOG).read_text(errors="replace")


def sampling_counters() -> Optional[Dict[str, Any]]:
    """The bridge's latest ``Motif-3 device sampling: {json}`` line (bridge + generator + sampler counters)."""
    txt = server_log_text()
    if txt is None:
        return None
    found = re.findall(r"Motif-3 device sampling: (\{.*\})", txt)
    return json.loads(found[-1]) if found else None


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


def salted(**kw) -> Dict[str, Any]:
    """A fresh ``cache_salt``: the request prefills cold (no prefix hit): its prefill numerics are run-independent."""
    return dict(kw, cache_salt=uuid.uuid4().hex)


# ======================================================================================================================
# tests
# ======================================================================================================================
def test_00_server_config(server):
    async def go(s):
        async with s.get(BASE + "/v1/models") as r:
            return r.status, await r.json()

    status, models = run(_session_do(go))
    txt = server_log_text() or ""
    budget = re.findall(r"Motif-3 features: chunked_prefill=\w+ \(budget (\d+), threshold (\d+)\)", txt)
    summary = dict(
        models=[m["id"] for m in models.get("data", [])],
        plugin_mode=bool(re.search(r"sample_on_device_mode=decode_only", txt)),
        bridge_sampler_on="Motif-3 device sampling: on" in txt,
        generator_sampler_on="device sampling on: K 64" in txt,
        features=(re.findall(r"Motif-3 features: .*", txt) or [None])[-1],
        budget=budget[-1] if budget else None,  # (budget, threshold) of the generator's feature line
        serving_config_warnings=re.findall(r"WARNING.*Motif-3 serving config.*", txt)[:5],
    )
    save("server_config", dict(summary=summary))
    assert status == 200 and MODEL in summary["models"]
    if SERVER_LOG:
        assert summary["plugin_mode"] and summary["bridge_sampler_on"] and summary["generator_sampler_on"], summary
        assert summary["features"] and f"spec_tokens={1 if SPEC else 0}" in summary["features"], summary["features"]
        want = os.environ.get("MOTIF3_DS_BUDGET")  # e.g. 8064: F5's budget = threshold, no misalignment warning
        if want:
            assert summary["budget"] == (want, want) and not summary["serving_config_warnings"], summary


async def _chat(s, pname: str, mode: str, thinking: Optional[bool], max_tokens: int) -> Dict[str, Any]:
    r = await stream(s, chat_body(SMOKE_PROMPTS[pname], max_tokens=max_tokens, thinking=thinking, **MODES[mode]))
    out = slim(r)
    out.update(prompt=pname, mode=mode, thinking=thinking, judge=judge(pname, answer_of(r)),
               degenerate=degenerate(r["token_ids"]))  # fmt: skip
    out["stopped_on_eos"] = r["finish_reason"] == "stop" and bool(r["token_ids"]) and r["token_ids"][-1] in EOS_IDS
    return out


def test_10_chats(server):
    async def go(s):
        t0 = time.perf_counter()
        a = await asyncio.gather(*[_chat(s, p, m, False, 512) for m in MODES for p in SMOKE_PROMPTS])
        w0 = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = await asyncio.gather(*[_chat(s, p, m, None, 3072) for m in MODES for p in SMOKE_PROMPTS])
        return list(a) + list(b), w0, time.perf_counter() - t0

    chats, w0, w1 = run(_session_do(go))
    ref = None
    if SMOKE_REF.is_file():
        d = json.loads(SMOKE_REF.read_text())
        ref = {(c["prompt"], c["mode"], c["thinking"]): c for c in d["chat"]}
    greedy = [c for c in chats if c["mode"] == "greedy"]
    by_g = {(c["prompt"], c["thinking"]): c["text"] for c in greedy}
    sampled = [c for c in chats if c["mode"] != "greedy"]
    g_same = [c for c in greedy if ref and (c["prompt"], "greedy", c["thinking"]) in ref
              and ref[(c["prompt"], "greedy", c["thinking"])]["text"] == c["text"]]  # fmt: skip
    per_mode = {m: dict(judged_ok=sum(bool(c["judge"].get("ok")) for c in chats if c["mode"] == m),
                        n=sum(c["mode"] == m for c in chats),
                        stopped_on_eos=sum(c["stopped_on_eos"] for c in chats if c["mode"] == m))
                for m in MODES}  # fmt: skip
    summary = dict(
        wall_s=[round(w0, 1), round(w1, 1)],
        status_ok=sum(c["status"] == 200 and not c["error"] for c in chats),
        per_mode=per_mode,
        degenerate=[(c["prompt"], c["mode"], c["thinking"]) for c in chats if c["degenerate"]],
        greedy_identical_to_draft1=f"{len(g_same)}/{len(greedy)}" if ref else None,
        sampled_equal_to_greedy=[(c["prompt"], c["mode"], c["thinking"]) for c in sampled
                                 if c["text"] == by_g[(c["prompt"], c["thinking"])]],  # fmt: skip
        samples={f"{c['prompt']}|{c['mode']}|{c['thinking']}": answer_of(c)[:300] for c in chats
                 if c["prompt"] in ("en_explain", "ko_question")},  # fmt: skip
    )
    save("chats", dict(summary=summary, chats=chats))
    assert summary["status_ok"] == len(chats), [c["error"] for c in chats if c["error"]]
    assert all(c["judge"].get("ok") for c in greedy), [(c["prompt"], c["thinking"]) for c in greedy
                                                       if not c["judge"]["ok"]]  # fmt: skip
    for m, v in per_mode.items():  # sampled answers: correct and well formed (T = 1 may miss a keyword now and then)
        assert v["judged_ok"] >= v["n"] - 2 and v["stopped_on_eos"] >= v["n"] - 1, (m, v)
    assert not summary["degenerate"], summary["degenerate"]
    assert len(summary["sampled_equal_to_greedy"]) <= 4, summary["sampled_equal_to_greedy"]
    if ref is not None:
        assert len(g_same) == len(greedy), "greedy texts differ from the draft-1 (host-sampled) server"


REPRO = [(QUESTIONS[i], dict(temperature=t, top_p=p, seed=1000 + i, **({"top_k": k} if k else {})))
         for i, (t, p, k) in enumerate([(1.0, 0.95, 0), (1.0, 1.0, 0), (0.6, 0.95, 0), (0.8, 1.0, 20), (1.2, 0.9, 0),
                                        (1.0, 1.0, 0), (0.7, 1.0, 0), (1.0, 0.95, 50), (1.0, 0.95, 0), (0.6, 1.0, 0),
                                        (1.0, 0.99, 0), (1.0, 1.0, 5)])]  # fmt: skip


def test_20_seeded_reproducible(server):
    async def one(s, q, kw, n=160):
        return await stream(s, chat_body(q, max_tokens=n, thinking=False, ignore_eos=True, **salted(**kw)))

    async def go(s):
        a = await asyncio.gather(*[one(s, q, kw) for q, kw in REPRO])
        b = []
        for q, kw in reversed(REPRO):  # one by one, reverse order: other lanes, other batches
            b.append(await one(s, q, kw))
        b.reverse()
        noise = [one(s, QUESTIONS[(i + 13) % len(QUESTIONS)], dict(temperature=1.0, top_p=0.95), 200)
                 for i in range(20)]  # fmt: skip
        c = await asyncio.gather(*([one(s, q, kw) for q, kw in REPRO[:6]] + noise))
        return list(a), b, list(c[:6])

    a, b, c = run(_session_do(go))
    same_ab = [x["token_ids"] == y["token_ids"] for x, y in zip(a, b)]
    same_ac = [x["token_ids"] == y["token_ids"] for x, y in zip(a[:6], c)]
    summary = dict(status_ok=sum(r["status"] == 200 and not r["error"] for r in a + b + c),
                   identical_concurrent_vs_sequential=f"{sum(same_ab)}/{len(same_ab)}",
                   identical_with_unseeded_noise=f"{sum(same_ac)}/{len(same_ac)}",
                   tokens=[len(r["token_ids"]) for r in a],
                   texts=[answer_of(r)[:160] for r in a])  # fmt: skip
    save("seeded_reproducible", dict(summary=summary, a=[slim(r) for r in a], b=[slim(r) for r in b]))
    assert summary["status_ok"] == len(a) + len(b) + len(c)
    assert all(same_ab) and all(same_ac), summary


def _mixed_cases() -> List[Dict[str, Any]]:
    cases = []
    for i in range(32):
        q = QUESTIONS[i % len(QUESTIONS)]
        kind = ["greedy", "topp", "full", "topk"][i % 4]
        if kind == "greedy":
            kw = dict(temperature=0.0)
        elif kind == "topp":
            kw = dict(temperature=1.0, top_p=0.95, seed=2000 + i)
        elif kind == "full":
            kw = dict(temperature=1.0 if i % 8 == 2 else 0.8, top_p=1.0, seed=2000 + i)
        else:
            kw = dict(temperature=1.0, top_p=1.0, top_k=20, seed=2000 + i)
        if not SPEC and i in (1, 9, 17, 25):  # logprobs=0 rows (a speculating launch refuses logprobs)
            kw.update(logprobs=True, top_logprobs=0)
        cases.append(dict(q=q, kind=kind, kw=kw))
    return cases


def test_30_mixed_batch(server):
    cases = _mixed_cases()

    async def one(s, c, n=200):
        return await stream(s, chat_body(c["q"], max_tokens=n, thinking=False, ignore_eos=True, **salted(**c["kw"])))

    async def go(s):
        solo = []
        for c in cases:  # the greedy rows alone, one by one
            if c["kind"] == "greedy":
                solo.append(await one(s, c))
        a = await asyncio.gather(*[one(s, c) for c in cases])
        b = await asyncio.gather(*[one(s, c) for c in cases])
        return solo, list(a), list(b)

    solo, a, b = run(_session_do(go))
    gi = [i for i, c in enumerate(cases) if c["kind"] == "greedy"]
    greedy_same = [solo[k]["token_ids"] == a[i]["token_ids"] for k, i in enumerate(gi)]
    seeded = [i for i, c in enumerate(cases) if c["kind"] != "greedy"]
    seeded_same = [a[i]["token_ids"] == b[i]["token_ids"] for i in seeded]
    lp_rows = [i for i, c in enumerate(cases) if c["kw"].get("logprobs")]
    lp_ok = all(
        len(a[i]["logprobs"]) == len(a[i]["token_ids"])
        and all(math.isfinite(x) and x <= 1e-6 for x in a[i]["logprobs"])
        for i in lp_rows
    )
    summary = dict(status_ok=sum(r["status"] == 200 and not r["error"] for r in solo + a + b),
                   greedy_equal_to_solo=f"{sum(greedy_same)}/{len(gi)}",
                   seeded_identical_across_batches=f"{sum(seeded_same)}/{len(seeded)}",
                   logprob_rows=len(lp_rows), logprobs_ok=lp_ok,
                   mean_logprob={i: round(sum(a[i]["logprobs"]) / max(1, len(a[i]["logprobs"])), 3) for i in lp_rows},
                   degenerate=[i for i, r in enumerate(a) if degenerate(r["token_ids"])])  # fmt: skip
    save("mixed_batch", dict(summary=summary, cases=cases, a=[slim(r) for r in a]))
    assert summary["status_ok"] == len(solo) + 2 * len(cases)
    assert all(greedy_same) and all(seeded_same) and lp_ok, summary
    assert not summary["degenerate"], summary


def test_40_host_routed(server):
    before = sampling_counters() or {}

    async def go(s):
        reqs = [
            stream(
                s,
                chat_body(
                    QUESTIONS[i],
                    max_tokens=96,
                    thinking=False,
                    ignore_eos=True,
                    **salted(temperature=1.0, top_p=0.95, seed=3000 + i),
                ),
            )
            for i in range(8)
        ]
        reqs.append(stream(s, chat_body(QUESTIONS[20], max_tokens=96, thinking=False, ignore_eos=True,
                                        **salted(temperature=0.0, presence_penalty=0.5))))  # fmt: skip
        return await asyncio.gather(*reqs)

    rs = run(_session_do(go))
    after = sampling_counters() or {}
    summary = dict(status_ok=sum(r["status"] == 200 and not r["error"] for r in rs),
                   host_steps_before=before.get("host_steps"), host_steps_after=after.get("host_steps"),
                   penalized_text=answer_of(rs[-1])[:200])  # fmt: skip
    save("host_routed", dict(summary=summary))
    assert summary["status_ok"] == len(rs)
    if SPEC:  # vLLM refuses no penalties on a speculating launch; PS-1 holds speculation back for them
        return
    if before and after:  # the counters come from the periodic log line (MOTIF3_SAMPLING_LOG_EVERY)
        assert after["host_steps"] >= before.get("host_steps", 0)


def test_50_counters(server):
    c = sampling_counters()
    if c is None:
        pytest.skip("no 'Motif-3 device sampling:' line in MOTIF3_DS_SERVER_LOG yet (MOTIF3_SAMPLING_LOG_EVERY)")
    save("counters", dict(summary=c))
    assert c["device_steps"] > 0 and c["sampled_steps"] > 0 and c["gumbel_lanes"] > 0, c
    assert c["fallback_step_rate"] < 0.10, c  # ~1-2 % of steps at T 1.0 / top-p 0.95 (DEVICE_SAMPLER.md §9)
    assert c["nongreedy_verify_rows"] == 0, c  # PS-1: no verify step held a sampled row


async def _tput_case(s, q: str, max_tokens: int, kw: Dict[str, Any]):
    return await stream(s, chat_body(q, max_tokens=max_tokens, thinking=None, ignore_eos=True, **salted(**kw)))


def test_60_tpot(server):
    levels = [int(x) for x in os.environ.get("MOTIF3_DS_TPUT_LEVELS", "1,8,32").split(",")]
    max_tokens = int(os.environ.get("MOTIF3_DS_TPUT_TOKENS", "256"))
    kinds = {"greedy": dict(temperature=0.0), "t1.0_p0.95": dict(temperature=1.0, top_p=0.95),
             "t1.0_p1": dict(temperature=1.0, top_p=1.0)}  # fmt: skip

    async def go(s):
        out = {}
        for c in levels:
            for kind, kw in kinds.items():
                m0 = await metrics(s)
                load0 = os.getloadavg()
                t0 = time.perf_counter()
                rs = await asyncio.gather(*[_tput_case(s, QUESTIONS[i % len(QUESTIONS)], max_tokens,
                                                       dict(kw, **({"seed": 4000 + i} if kw["temperature"] else {})))
                                            for i in range(c)])  # fmt: skip
                out[f"{kind}@{c}"] = dict(
                    raw=list(rs),
                    wall_s=time.perf_counter() - t0,
                    metrics=mdelta(await metrics(s), m0),
                    loadavg=[load0, os.getloadavg()],
                )
        return out

    res = run(_session_do(go))
    summary = {}
    for key, v in res.items():
        rs = v["raw"]
        summary[key] = dict(status_ok=sum(r["status"] == 200 and not r["error"] for r in rs),
                            tpot_ms=stats([1e3 * t for t in (tpot(r) for r in rs) if t]),
                            ttft_s=stats([r["ttft_s"] for r in rs]), drafts=v["metrics"]["spec_decode_num_drafts"],
                            acceptance=v["metrics"].get("acceptance"),
                            loadavg_1m=[round(x[0], 1) for x in v["loadavg"]])  # fmt: skip
    save("tpot", dict(summary=summary, runs={k: [slim(r) for r in v["raw"]] for k, v in res.items()}))
    for key, v in summary.items():
        assert v["status_ok"] == int(key.split("@")[1]), (key, v)
    if not SPEC:  # sampled ~ greedy: the same trace runs the sampler every step; only the rare host fallback adds
        for c in levels:
            g = summary[f"greedy@{c}"]["tpot_ms"]["p50"]
            for kind in ("t1.0_p0.95", "t1.0_p1"):
                s_ = summary[f"{kind}@{c}"]["tpot_ms"]["p50"]
                assert s_ <= 1.06 * g + 3.0, (kind, c, s_, g)


def test_70_greedy_speculation(server):
    if not SPEC:
        pytest.skip("the opt-in MTP launch only (dsamp_mtp)")

    async def go(s):
        m0 = await metrics(s)
        rs = await asyncio.gather(*[stream(s, chat_body(SMOKE_PROMPTS[p], max_tokens=512, thinking=False,
                                                        temperature=0.0)) for p in SMOKE_PROMPTS])  # fmt: skip
        return list(rs), mdelta(await metrics(s), m0)

    rs, md = run(_session_do(go))
    ref = load(REF_DIR, "chats")
    ref_g = {c["prompt"]: c["text"] for c in (ref or {}).get("chats", []) if c["mode"] == "greedy"
             and c["thinking"] is False}  # fmt: skip
    same = [r["text"] == ref_g.get(p) for p, r in zip(SMOKE_PROMPTS, rs)] if ref_g else None
    c = sampling_counters() or {}
    summary = dict(status_ok=sum(r["status"] == 200 and not r["error"] for r in rs), metrics=md,
                   identical_to_dsamp=None if same is None else f"{sum(same)}/{len(same)}",
                   verify_steps_with_params=c.get("verify_steps_with_params"),
                   nongreedy_verify_rows=c.get("nongreedy_verify_rows"))  # fmt: skip
    save("greedy_speculation", dict(summary=summary, texts=[r["text"] for r in rs]))
    assert summary["status_ok"] == len(rs)
    assert md["spec_decode_num_drafts"] > 0 and (md.get("acceptance") or 0) > 0.4, md
    if same is not None:
        assert all(same), summary


def test_80_device_vs_host_sampling(server):
    """The device sampler against vLLM's own host sampler on the same requests (``dsamp``: the speculating launch
    refuses logprobs): 8 prompts x 4 seeds at T 1.0 / top_p 1 (the Gumbel path) and T 1.0 / top-p 0.95, 128 tokens,
    ``logprobs=0``. Phase "device": as is; phase "host": the same requests with ``presence_penalty`` 1e-6 (a host-only
    parameter: the plugin samples those steps with vLLM's sampler; the penalty itself changes the logits by 1e-6). Both
    phases report the raw logprob of every sampled token; exact samplers of the same distributions give the same
    statistics: per-request mean logprob (Mann-Whitney / KS over the 32 requests), the token-level share of tail draws
    (raw logprob < -10, and < -6), the share of degenerate continuations."""
    if SPEC:
        pytest.skip("logprobs are refused on a speculating launch")
    from scipy import stats as sst

    modes = {"t1.0_p1": dict(temperature=1.0, top_p=1.0), "t1.0_p0.95": dict(temperature=1.0, top_p=0.95)}
    qs = QUESTIONS[:8]

    async def phase(s, kw, host):
        extra = dict(presence_penalty=1e-6) if host else {}
        reqs = [stream(s, chat_body(q, max_tokens=128, thinking=False, ignore_eos=True, logprobs=True, top_logprobs=0,
                                    **salted(**kw, seed=5000 + 17 * i + j, **extra)))
                for i, q in enumerate(qs) for j in range(4)]  # fmt: skip
        return list(await asyncio.gather(*reqs))

    async def go(s):
        out = {}
        for name, kw in modes.items():
            out[name] = {"device": await phase(s, kw, False), "host": await phase(s, kw, True)}
        return out

    before = sampling_counters() or {}
    res = run(_session_do(go))
    after = sampling_counters() or {}
    summary = {"host_steps_delta": (after.get("host_steps", 0) - before.get("host_steps", 0)) if after else None}
    for name, ph in res.items():
        row = {}
        for side, rs in ph.items():
            lps = [r["logprobs"][1:] for r in rs]  # the first token is the prefill's (host-sampled in both phases)
            flat = [x for lp in lps for x in lp]
            row[side] = dict(status_ok=sum(r["status"] == 200 and not r["error"] for r in rs),
                             req_mean=[sum(lp) / max(1, len(lp)) for lp in lps], tokens=len(flat),
                             tail10=sum(x < -10 for x in flat) / max(1, len(flat)),
                             tail6=sum(x < -6 for x in flat) / max(1, len(flat)),
                             mean=sum(flat) / max(1, len(flat)),
                             degenerate=sum(degenerate(r["token_ids"], n=4, limit=8) for r in rs))  # fmt: skip
        d, h = row["device"], row["host"]
        k6 = [round(d["tail6"] * d["tokens"]), round(h["tail6"] * h["tokens"])]
        tab = [[k6[0], d["tokens"] - k6[0]], [k6[1], h["tokens"] - k6[1]]]
        summary[name] = dict(
            device_mean=round(d["mean"], 4), host_mean=round(h["mean"], 4),
            mannwhitney_p=float(sst.mannwhitneyu(d["req_mean"], h["req_mean"]).pvalue),
            ks_p=float(sst.ks_2samp(d["req_mean"], h["req_mean"]).pvalue),
            tail10=[round(d["tail10"], 4), round(h["tail10"], 4)], tail6=[round(d["tail6"], 4), round(h["tail6"], 4)],
            tail6_fisher_p=float(sst.fisher_exact(tab).pvalue), degenerate=[d["degenerate"], h["degenerate"]],
            status_ok=[d["status_ok"], h["status_ok"]], tokens=[d["tokens"], h["tokens"]],
        )  # fmt: skip
    raw = {k: {side: [slim(r) for r in v[side]] for side in v} for k, v in res.items()}
    save("device_vs_host", dict(summary=summary, raw=raw))
    for name in modes:
        v = summary[name]
        assert v["status_ok"] == [32, 32], v
        assert v["mannwhitney_p"] > 1e-3 and v["ks_p"] > 1e-3, (name, v)
        assert v["tail6_fisher_p"] > 1e-3, (name, v)
    # summary["host_steps_delta"]: the bridge's host-routed step count (informational: the counter line is periodic)
