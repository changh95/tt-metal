# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the P/D bundle proxy (``server/pd_app.py``): the fan-out of a generation request to P and D
at the same time under a proxy-chosen ``transfer_id``, and what happens when P fails or does not echo it.

The two vLLM halves are an ``httpx.MockTransport``; nothing is launched and no device is touched.  The app's
lifespan (which launches ``vllm serve`` on the chips) must never run here: the TestClient is used without its
context manager and ``VllmHalf.start`` is disabled for every test.
"""

import asyncio
import dataclasses
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


def test_weight_cache_is_warm_only_with_a_marker_for_this_code(tmp_path):
    """An existing cache directory from an older image is cold (new code may add weight files; two halves converting
    the same missing files at once corrupt each other); the marker written at READY makes it warm."""
    cache = tmp_path / "cache"
    (cache / "P150x4" / "tensor_cache_bfp8_mesh1x4" / "layers.0").mkdir(parents=True)
    (cache / "P150x4" / "tensor_cache_bfp8_mesh1x4" / "layers.0" / "x.tensorbin").write_bytes(b"1")
    env = {"TT_CACHE_PATH": str(cache)}
    assert pd_app.weight_cache_is_warm(env) is False
    pd_app.mark_weight_cache_warm(env)
    assert pd_app.weight_cache_is_warm(env) is True
    assert (cache / f".qwen36_pd_warm_{pd_app.model_code_fingerprint()}").is_file()
    assert pd_app.weight_cache_is_warm({"TT_CACHE_PATH": str(tmp_path / "missing")}) is False
    assert pd_app.weight_cache_is_warm({}) is False


def test_vllm_argv_speculative_mtp_only_on_the_decode_half():
    """QWEN36_SPEC_MTP=1 adds vLLM's speculative_config (MTP, QWEN36_SPEC_K drafts) and --no-async-scheduling to the
    decode half only; the prefill half's argv is unchanged (it reads the env itself for the MTP prefill/export)."""
    base = {"HF_MODEL": "Qwen/Qwen3.8-27B", "HF_HUB_OFFLINE": "1"}
    off = pd_app.BundleConfig.from_env(base)
    on = pd_app.BundleConfig.from_env({**base, "QWEN36_SPEC_MTP": "1", "QWEN36_SPEC_K": "2"})
    assert off.speculative_mtp is False and on.speculative_mtp is True and on.speculative_k == 2
    assert (
        pd_app.vllm_argv(off, "decode")
        == [a for a in pd_app.vllm_argv(on, "decode") if a not in ("--no-async-scheduling",)][
            : len(pd_app.vllm_argv(off, "decode"))
        ]
    )
    argv = pd_app.vllm_argv(on, "decode")
    i = argv.index("--speculative-config")
    assert argv[i + 1] == '{"method": "mtp", "num_speculative_tokens": 2}'
    assert "--no-async-scheduling" in argv
    assert pd_app.vllm_argv(on, "prefill") == pd_app.vllm_argv(off, "prefill")


def test_dflash2_drafter_config_and_weights_resolution(tmp_path, monkeypatch, expect_error):
    """QWEN36_SPEC_DRAFTER=dflash2: the draft count defaults to the block's 7, the halves get the knob and the resolved
    drafter snapshot (DFLASH2_MODEL) in their env, a snapshot directory is taken as is, and a missing cache offline
    is a BundleConfigError naming the download; mtp / plain configs are unchanged."""
    base = {"HF_MODEL": "Qwen/Qwen3.8-27B", "HF_HUB_OFFLINE": "1", "QWEN36_SPEC_MTP": "1"}
    df = pd_app.BundleConfig.from_env({**base, "QWEN36_SPEC_DRAFTER": "dflash2"})
    assert df.uses_dflash2 and df.speculative_k == 7 and df.dflash2_model == pd_app.DFLASH2_REPO
    assert df.dflash2_revision == pd_app.DFLASH2_REVISION
    argv = pd_app.vllm_argv(df, "decode")
    assert argv[argv.index("--speculative-config") + 1] == '{"method": "mtp", "num_speculative_tokens": 7}'
    mtp = pd_app.BundleConfig.from_env(base)
    assert not mtp.uses_dflash2 and mtp.speculative_k == 3 and mtp.speculative_drafter == "mtp"
    assert (
        pd_app.BundleConfig.from_env({**base, "QWEN36_SPEC_DRAFTER": "dflash2", "QWEN36_SPEC_K": "3"}).speculative_k
        == 3
    )
    with expect_error(pd_app.BundleConfigError, "QWEN36_SPEC_DRAFTER"):
        pd_app.BundleConfig.from_env({**base, "QWEN36_SPEC_DRAFTER": "eagle"})
    # env of the halves: the knob on both, the snapshot only once resolved
    for role in ("prefill", "decode"):
        env = pd_app.vllm_env(df, role, base={})
        assert env["QWEN36_SPEC_MTP"] == "1" and env["QWEN36_SPEC_DRAFTER"] == "dflash2" and "DFLASH2_MODEL" not in env
        assert pd_app.vllm_env(mtp, role, base={})["QWEN36_SPEC_DRAFTER"] == "mtp"
        assert "QWEN36_SPEC_MTP" not in pd_app.vllm_env(
            pd_app.BundleConfig.from_env({**base, "QWEN36_SPEC_MTP": "0"}), role, base={}
        )
    snap = tmp_path / "dflash2"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    local = pd_app.BundleConfig.from_env({**base, "QWEN36_SPEC_DRAFTER": "dflash2", "DFLASH2_MODEL": str(snap)})
    assert pd_app.resolve_dflash2_snapshot(local) == str(snap)
    resolved = dataclasses.replace(local, dflash2_snapshot=pd_app.resolve_dflash2_snapshot(local))
    assert pd_app.vllm_env(resolved, "decode", base={})["DFLASH2_MODEL"] == str(snap)
    desc = pd_app.bundle_description(resolved, None)["speculative"]
    assert desc["drafter"] == "dflash2" and desc["max_drafts_per_step"] == 7 and desc["drafter_weights"] == str(snap)
    assert pd_app.bundle_description(mtp, None)["speculative"] == {"drafter": "mtp", "max_drafts_per_step": 3}
    # offline with the snapshot missing from the cache (the hub raises): the error names the repo, the revision and the
    # download command
    import huggingface_hub

    def missing(repo_id, revision=None, local_files_only=False, **kw):
        assert (repo_id, revision, local_files_only) == (pd_app.DFLASH2_REPO, pd_app.DFLASH2_REVISION, True)
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with expect_error(pd_app.BundleConfigError, "huggingface-cli download z-lab/Qwen3.8-27B-DFlash2"):
        pd_app.resolve_dflash2_snapshot(df)


def test_hybrid_drafter_config(tmp_path):
    """QWEN36_SPEC_DRAFTER=hybrid: both drafters resident on D (the DFlash2 weights repo is resolved like dflash2's),
    the draft count defaults to the block's 7 (the hybrid's DFlash2 band; its MTP bands clamp their own T), both halves
    get the knob, and the description names the policy."""
    base = {"HF_MODEL": "Qwen/Qwen3.8-27B", "HF_HUB_OFFLINE": "1", "QWEN36_SPEC_MTP": "1"}
    hy = pd_app.BundleConfig.from_env({**base, "QWEN36_SPEC_DRAFTER": "hybrid"})
    assert hy.uses_dflash2 and hy.speculative_drafter == "hybrid" and hy.speculative_k == 7
    argv = pd_app.vllm_argv(hy, "decode")
    assert argv[argv.index("--speculative-config") + 1] == '{"method": "mtp", "num_speculative_tokens": 7}'
    assert "--speculative-config" not in pd_app.vllm_argv(hy, "prefill")
    for role in ("prefill", "decode"):
        env = pd_app.vllm_env(hy, role, base={})
        assert env["QWEN36_SPEC_MTP"] == "1" and env["QWEN36_SPEC_DRAFTER"] == "hybrid" and "DFLASH2_MODEL" not in env
    snap = tmp_path / "dflash2"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    local = pd_app.BundleConfig.from_env({**base, "QWEN36_SPEC_DRAFTER": "hybrid", "DFLASH2_MODEL": str(snap)})
    resolved = dataclasses.replace(local, dflash2_snapshot=pd_app.resolve_dflash2_snapshot(local))
    assert pd_app.vllm_env(resolved, "decode", base={})["DFLASH2_MODEL"] == str(snap)
    desc = pd_app.bundle_description(resolved, None)["speculative"]
    assert desc == {
        "drafter": "hybrid",
        "max_drafts_per_step": 7,
        "drafter_weights": str(snap),
        "revision": pd_app.DFLASH2_REVISION,
    }
    # the sibling knobs are unchanged
    assert pd_app.BundleConfig.from_env({**base, "QWEN36_SPEC_DRAFTER": "mtp"}).speculative_k == 3
    assert not pd_app.BundleConfig.from_env(
        {**base, "QWEN36_SPEC_MTP": "0", "QWEN36_SPEC_DRAFTER": "hybrid"}
    ).uses_dflash2


def test_kv_pool_is_per_role():
    """QWEN36_MAX_TOKENS_ALL_USERS (each vLLM's KV pool) is set per half: the prefill half keeps the p300x2 spec's
    525,312 tokens (it only holds in-flight prefills), the decode half gets QWEN36_PD_DECODE_MAX_TOKENS (default
    pd_app.DECODE_MAX_TOKENS: 32 x 32k / 16 x 64k / 8 x 128k live contexts); an operator's own value never leaks from
    one role to the other, and the bundle description reports both."""
    assert "QWEN36_MAX_TOKENS_ALL_USERS" not in pd_app.CHILD_ENV
    base = {"HF_MODEL": "Qwen/Qwen3.8-27B", "QWEN36_MAX_TOKENS_ALL_USERS": "1"}
    cfg = pd_app.BundleConfig.from_env(base)
    assert cfg.prefill_max_tokens == pd_app.PREFILL_MAX_TOKENS == 525312
    assert cfg.decode_max_tokens == pd_app.DECODE_MAX_TOKENS
    assert pd_app.DECODE_MAX_TOKENS % 64 == 0 and pd_app.DECODE_MAX_TOKENS >= 32 * (32768 + 128)
    assert pd_app.DECODE_MAX_TOKENS >= 16 * (65536 + 128) and pd_app.DECODE_MAX_TOKENS >= 8 * (131072 + 128)
    p_env = pd_app.vllm_env(cfg, "prefill", base)
    d_env = pd_app.vllm_env(cfg, "decode", base)
    assert p_env["QWEN36_MAX_TOKENS_ALL_USERS"] == "525312"
    assert d_env["QWEN36_MAX_TOKENS_ALL_USERS"] == str(pd_app.DECODE_MAX_TOKENS)
    custom = pd_app.BundleConfig.from_env(
        {**base, "QWEN36_PD_DECODE_MAX_TOKENS": "1048576", "QWEN36_PD_PREFILL_MAX_TOKENS": "262144"}
    )
    assert pd_app.vllm_env(custom, "decode", base)["QWEN36_MAX_TOKENS_ALL_USERS"] == "1048576"
    assert pd_app.vllm_env(custom, "prefill", base)["QWEN36_MAX_TOKENS_ALL_USERS"] == "262144"
    stack = pd_app.Stack(cfg, pd_app.VllmHalf(cfg, "prefill"), pd_app.VllmHalf(cfg, "decode"))
    halves = pd_app.bundle_description(cfg, stack)["halves"]
    assert (
        halves["prefill"]["kv_pool_tokens"] == 525312 and halves["decode"]["kv_pool_tokens"] == pd_app.DECODE_MAX_TOKENS
    )
