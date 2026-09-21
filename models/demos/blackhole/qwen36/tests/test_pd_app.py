# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the P/D bundle proxy (``server/pd_app.py``): the fan-out of a generation request to P and D
at the same time under a proxy-chosen ``transfer_id``, and what happens when P fails or does not echo it.

The two vLLM halves are an ``httpx.MockTransport``; nothing is launched and no device is touched.  The app's
lifespan (which launches ``vllm serve`` on the chips) must never run here: the TestClient is used without its
context manager and ``VllmHalf.start`` is disabled for every test.
"""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from models.demos.blackhole.qwen36.server import pd_app

CONFIG = pd_app.BundleConfig.from_env({"QWEN36_PD_SIDE_PORT": "18111"})
SERIAL = pd_app.BundleConfig.from_env({"QWEN36_PD_PROXY_FANOUT": "0", "QWEN36_PD_SIDE_PORT": "18111"})
BODY = {"model": pd_app.MODEL_ID, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "stream": True}
SSE = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'


def _p_params(transfer_id):
    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_host": "127.0.0.1",
        "remote_port": CONFIG.side_channel_port,
        "transfer_id": transfer_id,
        "num_tokens": 5,
    }


class Halves:
    """The mocked P and D.  P answers only once D has been posted to (bounded), so the recorded order proves the
    fan-out; ``d_holds`` makes D hang like a real decoder waiting for the transfer, to observe its cancellation."""

    def __init__(self, p_status=200, echo=True, d_holds=False):
        self.p_status, self.echo, self.d_holds = p_status, echo, d_holds
        self.events = []
        self._d_arrived = None

    def d_arrived(self):
        if self._d_arrived is None:
            self._d_arrived = asyncio.Event()
        return self._d_arrived

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        params = body.get("kv_transfer_params")
        if request.url.port == CONFIG.prefill_port:
            self.events.append(("P", params, dict(request.headers), body))
            try:
                await asyncio.wait_for(self.d_arrived().wait(), 0.5)
            except asyncio.TimeoutError:
                self.events.append(("P answered before D was posted to",))
            if self.p_status != 200:
                return httpx.Response(self.p_status, json={"error": {"message": "too long", "type": "invalid"}})
            tid = params["transfer_id"] if self.echo else "chatcmpl-engine-id-1a2b3c4d"
            return httpx.Response(
                200, json={"choices": [{"message": {"content": ""}}], "kv_transfer_params": _p_params(tid)}
            )
        self.events.append(("D", params, dict(request.headers), body))
        self.d_arrived().set()
        if self.d_holds:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.events.append(("D cancelled",))
                raise
        # an unread stream, like a live decoder's: the proxy relays it chunk by chunk
        return httpx.Response(200, stream=httpx.ByteStream(SSE), headers={"content-type": "text/event-stream"})


@pytest.fixture(autouse=True)
def never_launch_vllm(monkeypatch):
    def refuse(self):
        raise RuntimeError("test_pd_app must not launch vLLM (the lifespan ran?)")

    monkeypatch.setattr(pd_app.VllmHalf, "start", refuse)


def _client(halves, config=CONFIG):
    app = pd_app.create_app()
    app.state.config = config
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(halves))
    app.state.stack = pd_app.Stack(config, pd_app.VllmHalf(config, "prefill"), pd_app.VllmHalf(config, "decode"))
    # never `with TestClient(...)`: the context manager runs the lifespan, which launches vLLM
    return TestClient(app)


def test_fanout_posts_to_d_while_p_prefills():
    halves = Halves()
    r = _client(halves).post("/v1/chat/completions", json=BODY, headers={"X-Request-Id": "req-1"})
    assert r.status_code == 200 and r.content == SSE
    kinds = [e[0] for e in halves.events]
    assert kinds == ["P", "D"] or kinds == ["D", "P"]  # both in flight before P answered
    assert ("P answered before D was posted to",) not in halves.events
    (p,) = [e for e in halves.events if e[0] == "P"]
    (d,) = [e for e in halves.events if e[0] == "D"]
    tid = p[1]["transfer_id"]
    assert tid.startswith("pd-") and p[1]["do_remote_decode"] is True
    assert d[1] == pd_app.fanout_transfer_params(CONFIG, tid)
    assert "num_tokens" not in d[1]  # D derives it from its own tokenization
    assert p[3]["max_tokens"] == 1 and p[3]["stream"] is False
    assert d[3]["max_tokens"] == 8 and d[3]["stream"] is True
    assert p[2]["x-request-id"] == d[2]["x-request-id"] == "req-1"


def test_p_failure_returns_p_error_and_cancels_d():
    halves = Halves(p_status=400, d_holds=True)
    r = _client(halves).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 400 and r.json()["error"]["message"] == "too long"
    assert ("D cancelled",) in halves.events
    assert [e[0] for e in halves.events if e[0] == "D"] == ["D"]  # never re-posted


def test_missing_echo_falls_back_to_the_serial_round_trip():
    halves = Halves(echo=False, d_holds=True)
    r = _client(halves).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200 and r.content == SSE
    d_posts = [e for e in halves.events if e[0] == "D"]
    assert len(d_posts) == 2 and ("D cancelled",) in halves.events
    # the second D post carries P's own parameters (engine request id, num_tokens)
    assert d_posts[1][1] == _p_params("chatcmpl-engine-id-1a2b3c4d")


def test_serial_mode_posts_to_d_after_p():
    halves = Halves()
    r = _client(halves, SERIAL).post(
        "/v1/completions", json={"model": pd_app.MODEL_ID, "prompt": "hi", "max_tokens": 4}
    )
    assert r.status_code == 200
    assert [e[0] for e in halves.events] == ["P", "P answered before D was posted to", "D"]
    p, d = [e for e in halves.events if e[0] in ("P", "D")]
    assert d[1] == _p_params(p[1]["transfer_id"])  # P's params verbatim, transfer_id still the proxy's


def test_pure_helpers():
    t1, t2 = pd_app.new_transfer_id(), pd_app.new_transfer_id()
    assert t1 != t2 and t1.startswith("pd-")
    p = pd_app.prefill_request({"stream": True, "max_tokens": 9, "min_tokens": 2, "stream_options": {}}, "t")
    assert p["kv_transfer_params"]["transfer_id"] == "t" and p["max_tokens"] == 1 and p["stream"] is False
    assert "min_tokens" not in p and "stream_options" not in p
    assert "transfer_id" not in pd_app.prefill_request({})["kv_transfer_params"]
    d = pd_app.decode_request({"prompt": "x"}, {"transfer_id": "t"})
    assert d == {"prompt": "x", "kv_transfer_params": {"transfer_id": "t"}}
    desc = pd_app.bundle_description(CONFIG, None)
    assert desc["topology"]["proxy"].startswith("fan-out")
    assert pd_app.bundle_description(SERIAL, None)["topology"]["proxy"].startswith("serial")
    assert isinstance(pd_app.unlink_stale_shm_segments(), list)


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False), ("false", False), ("", True)])
def test_fanout_knob(value, expected):
    env = {"QWEN36_PD_PROXY_FANOUT": value} if value else {}
    assert pd_app.BundleConfig.from_env(env).proxy_fanout is expected
