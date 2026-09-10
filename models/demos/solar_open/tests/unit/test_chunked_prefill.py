# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the chunked single-user prefill (phase 3d, ``tt/chunked_prefill.py``) -- no device is opened.

``tt/chunked_prefill.py`` imports no ttnn (it can be loaded by file path with importlib on a box whose devices are
busy; that is how the B1 stage exercised it). Under pytest the package imports below are fine (importing ttnn opens
no device):

  * the ``SOLAR_OPEN_PREFILL_CHUNK_TOKENS`` knob: default 32768, validation (a power of two in 2048..131072), and
    ``ModelArgs.max_prefill_chunk_size`` reading it without disturbing the traced-prefill / warm-up tables;
  * ``chunk_schedule``: the chunk starts, page-table block ranges, ``get_last_token`` rows and the early return the
    Generator's ``prefill_forward_single_user_text`` produces (cross-checked against the Generator's own
    ``get_max_prefill_chunk_size``), every start aligned to the page block and the SDPA chunk sizes of
    ``SolarOpenAttentionProgramConfig``;
  * ``rope_slice_bounds`` (the cos / sin rows of a chunk, bounded by the 131072-position tables) and the demo's
    ``single_row_prefill_cap``.

    SOLAR_OPEN_NUM_DEVICES=8 pytest models/demos/solar_open/tests/unit/test_chunked_prefill.py
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.demos.solar_open.tt import chunked_prefill as cp
from models.demos.solar_open.tt import model_config as mc
from models.demos.solar_open.tt.attention_configs import SolarOpenAttentionProgramConfig
from models.tt_transformers.tt.common import get_max_prefill_chunk_size, get_padded_prefill_len

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "Solar-Open-100B"
BLOCK = 64


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv(cp.PREFILL_CHUNK_TOKENS_ENV, raising=False)
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path / "tt_cache"))
    monkeypatch.setattr(mc, "determine_device_name", lambda mesh_device: "P150x8")


@pytest.fixture
def mesh_1x8():
    mesh = MagicMock(name="mesh_device_1x8")
    mesh.shape = (1, 8)
    return mesh


class TestKnob:
    def test_default_and_env(self, monkeypatch):
        assert cp.prefill_chunk_tokens_from_env() == cp.PREFILL_CHUNK_TOKENS_DEFAULT == 32 * 1024
        monkeypatch.setenv(cp.PREFILL_CHUNK_TOKENS_ENV, "")
        assert cp.prefill_chunk_tokens_from_env() == 32 * 1024
        for value in ("2048", " 4096 ", "65536", "131072"):
            monkeypatch.setenv(cp.PREFILL_CHUNK_TOKENS_ENV, value)
            assert cp.prefill_chunk_tokens_from_env() == int(value)

    @pytest.mark.parametrize("value", ["1000", "0", "-4096", "abc", "3072", "1024", "262144", "24576"])
    def test_rejects(self, monkeypatch, value, expect_error):
        monkeypatch.setenv(cp.PREFILL_CHUNK_TOKENS_ENV, value)
        with expect_error(ValueError, cp.PREFILL_CHUNK_TOKENS_ENV):
            cp.prefill_chunk_tokens_from_env()

    def test_model_args_reads_the_knob(self, monkeypatch, mesh_1x8, expect_error):
        monkeypatch.setenv("HF_MODEL", str(CONFIG_DIR))
        args = mc.ModelArgs(mesh_device=mesh_1x8, dummy_weights=True)
        assert args.max_prefill_chunk_size == 32 * 1024
        # The traced-prefill / warm-up tables do not depend on the chunk (128 << 32K; warm-up stops at 2048).
        assert args.trace_prefill_supported_seq_lens == [128] and args.can_enable_trace(128)
        assert not args.can_enable_trace(128, num_cached_tokens=32)
        assert args.get_warmup_prefill_supported_seq_lens() == [128, 1024, 2048]
        monkeypatch.setenv(cp.PREFILL_CHUNK_TOKENS_ENV, "65536")
        assert mc.ModelArgs(mesh_device=mesh_1x8, dummy_weights=True).max_prefill_chunk_size == 64 * 1024
        monkeypatch.setenv(cp.PREFILL_CHUNK_TOKENS_ENV, "131072")  # never chunk: the phase-3c value
        assert mc.ModelArgs(mesh_device=mesh_1x8, dummy_weights=True).max_prefill_chunk_size == 128 * 1024
        monkeypatch.setenv(cp.PREFILL_CHUNK_TOKENS_ENV, "1000")
        with expect_error(ValueError, cp.PREFILL_CHUNK_TOKENS_ENV):
            mc.ModelArgs(mesh_device=mesh_1x8, dummy_weights=True)

    def test_single_row_cap(self, expect_error):
        assert cp.single_row_prefill_cap(2048) == cp.single_row_prefill_cap(32768) == 131072
        assert cp.single_row_prefill_cap(65536) == 131072  # a 64K single pass is the validated cap itself
        assert cp.single_row_prefill_cap(131072) == 65536  # never chunk -> the phase-3c 64K cap
        with expect_error(ValueError, "power of two"):
            cp.single_row_prefill_cap(1000)


class TestSchedule:
    def test_128k_in_32k_chunks(self):
        # prefill_128k: the 128K prompt is clipped to 131072 - 200 = 130872 tokens (padded 131072)
        s = cp.chunk_schedule(131072, 32768, BLOCK, 130871)
        assert [c.start for c in s] == [0, 32768, 65536, 98304]
        assert [c.end for c in s] == [32768, 65536, 98304, 131072]
        assert [(c.block_start, c.block_end) for c in s] == [(0, 512), (512, 1024), (1024, 1536), (1536, 2048)]
        assert [c.is_last for c in s] == [False, False, False, True]
        # the Generator hands every chunk the last-token row offset of the LAST chunk (32-row tile)
        assert {c.get_last_token for c in s} == {((130871 - 98304) // 32) * 32}
        assert [c.chunk_start_idx for c in s] == [None, 32768, 65536, 98304]  # chunk 0: legacy path

    def test_early_return_after_the_last_token_chunk(self):
        s = cp.chunk_schedule(131072, 32768, BLOCK, 69999)  # a 70K prompt padded to 128K
        assert [c.start for c in s] == [0, 32768, 65536] and s[-1].is_last
        assert s[-1].get_last_token == ((69999 - 65536) // 32) * 32

    def test_device_test_shapes(self):
        # tests/test_chunked_prefill.py: 7000 tokens padded to 8192, 4096- and 2048-token chunks
        assert get_padded_prefill_len(7000) == 8192
        s4 = cp.chunk_schedule(8192, 4096, BLOCK, 6999)
        assert [c.start for c in s4] == [0, 4096] and [c.block_end - c.block_start for c in s4] == [64, 64]
        assert s4[-1].get_last_token == ((6999 - 4096) // 32) * 32 == 2880
        s2 = cp.chunk_schedule(8192, 2048, BLOCK, 6999)
        assert [c.start for c in s2] == [0, 2048, 4096, 6144] and s2[-1].get_last_token == ((6999 - 6144) // 32) * 32

    def test_unchunked_prefills_are_one_legacy_pass(self):
        for seq_len, last in ((128, 100), (1024, 1023), (32768, 20000), (32768, 32767)):
            (only,) = cp.chunk_schedule(seq_len, 32768, BLOCK, last)
            assert only.start == 0 and only.end == seq_len and only.chunk_start_idx is None and only.is_last
            assert only.block_end == -(-seq_len // BLOCK) and only.get_last_token == (last // 32) * 32
        (only,) = cp.chunk_schedule(131072, 131072, BLOCK, 130871)  # env 131072 = never chunk
        assert only.end == 131072 and only.chunk_start_idx is None

    def test_resumed_form_mirrors_the_generator(self):
        # a suffix of 4096 tokens after 2048 cached ones (the Generator's prefix-caching form; unused by Solar today)
        (only,) = cp.chunk_schedule(4096, 32768, BLOCK, 5000, num_cached_tokens=2048)
        assert (only.start, only.end, only.block_start, only.block_end) == (2048, 6144, 32, 96)
        assert only.chunk_start_idx == 2048 and only.get_last_token == ((5000 - 2048) // 32) * 32

    def test_chunk_size_matches_the_generator(self):
        for seq_len in (2048, 4096, 8192, 16384, 32768, 65536, 131072):
            for cap in (2048, 4096, 8192, 16384, 24576, 32768, 65536, 131072):
                assert cp.generator_chunk_size(seq_len, cap) == get_max_prefill_chunk_size(seq_len, cap)

    def test_every_start_is_aligned(self):
        pc = SolarOpenAttentionProgramConfig()
        assert cp.SDPA_CHUNK_ALIGN == pc.prefill_q_chunk_size_large == pc.prefill_k_chunk_size_large
        assert pc.prefill_threshold <= cp.PREFILL_CHUNK_TOKENS_MIN  # every chunk uses the large SDPA chunk sizes
        for chunk in (2048, 4096, 32768, 65536):
            for c in cp.chunk_schedule(131072, chunk, BLOCK, 131071):
                assert c.start % cp.SDPA_CHUNK_ALIGN == 0 and c.start % BLOCK == 0 and c.end - c.start == chunk

    @pytest.mark.parametrize(
        "args, message",
        [
            ((8192, 4096, BLOCK, 8192), "last_token_idx"),
            ((8192, 4096, BLOCK, -1), "last_token_idx"),
            ((0, 4096, BLOCK, 0), "must be positive"),
            ((8192, 4096, 0, 10), "must be positive"),
        ],
    )
    def test_rejects(self, args, message, expect_error):
        with expect_error(ValueError, message):
            cp.chunk_schedule(*args)


class TestRopeBounds:
    def test_bounds(self, expect_error):
        assert cp.rope_slice_bounds(0, 128, 131072) == (0, 128)
        assert cp.rope_slice_bounds(98304, 32768, 131072) == (98304, 131072)
        with expect_error(ValueError, "RoPE tables"):
            cp.rope_slice_bounds(98304, 65536, 131072)
        with expect_error(ValueError, "start_pos >= 0 and seq_len > 0"):
            cp.rope_slice_bounds(-1, 128, 131072)
        with expect_error(ValueError, "start_pos >= 0 and seq_len > 0"):
            cp.rope_slice_bounds(0, 0, 131072)
