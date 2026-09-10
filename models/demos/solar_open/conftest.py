# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures shared by every Solar-Open test and demo (unit tests, integration tests, text demo)."""

import json
import os
from pathlib import Path

import pytest
from loguru import logger

# PCC thresholds per component, keyed by ModelArgs.model_name ("Solar-Open-100B") and mode (decode/prefill).
# Resolved relative to this file so the fixture works from any working directory.
UNIT_TEST_THRESHOLDS_PATH = Path(__file__).resolve().parent / "unit_test_thresholds.json"

# Calendar day the recorded chat-templated references of this port were tokenized on (README "Recorded baselines":
# the real-weight layer-0 digits, the packed-prefill floors of tests/test_layer0_batched_prefill.py and
# tests/unit/test_batched_prefill.py, the multi-user consistency counts, the demo token counts). Solar's chat template
# stamps ``strftime_now("%Y-%m-%d")`` into its provider system prompt, so a test that tokenizes prompts reproduces its
# recorded digits only with this date pinned (tt/model_config.py::template_date_kwargs, env SOLAR_OPEN_TEMPLATE_DATE).
# The teacher-forced test pins its own reference date and does not use this fixture.
RECORDED_TEMPLATE_DATE = "2026-09-08"
TEMPLATE_DATE_ENV = (
    "SOLAR_OPEN_TEMPLATE_DATE"  # == tt/model_config.py::TEMPLATE_DATE_ENV (asserted in test_model_config)
)


def pytest_addoption(parser):
    parser.addoption("--skip-model-load", action="store_true", default=False, help="Skip loading the model state dict")


@pytest.fixture(scope="session")
def state_dict(request):
    load_model = not request.config.getoption("--skip-model-load")
    model_path = os.getenv("HF_MODEL", None)
    if model_path is None or not load_model:
        # Explicit skip: build weights purely from the ttnn cache.
        return {}
    # Defer the (expensive) HF weight load to create_tt_model, which knows the mesh shape +
    # dtype and can skip it when a warm ttnn cache is already on disk (see
    # ModelArgs.weight_cache_is_complete). Returning None signals "load if needed".
    return None


@pytest.fixture
def pinned_template_date(monkeypatch):
    """Pin the chat template's date to ``RECORDED_TEMPLATE_DATE`` for this test (``ModelArgs.encode_prompt`` and the
    ``template_date_kwargs()`` callers read the variable at call time).

    An exported ``SOLAR_OPEN_TEMPLATE_DATE`` wins: ``=2026-09-09`` reproduces that day's token ids, ``=today`` (or
    empty) un-pins -- the production rendering with the real date. Returns the effective value ("today" when
    un-pinned) so a test can log which ids it ran on.
    """
    if TEMPLATE_DATE_ENV not in os.environ:
        monkeypatch.setenv(TEMPLATE_DATE_ENV, RECORDED_TEMPLATE_DATE)
    date = os.environ[TEMPLATE_DATE_ENV].strip() or "today"
    logger.info(
        f"chat template date: {TEMPLATE_DATE_ENV}={date}"
        + (
            " (recorded-reference date)"
            if date == RECORDED_TEMPLATE_DATE
            else " (exported override; digits may differ)"
        )
    )
    return date


@pytest.fixture
def test_thresholds(request):
    """Component PCC thresholds: ``thresholds["Solar-Open-100B"][mode][component]``."""
    with open(UNIT_TEST_THRESHOLDS_PATH, "r") as f:
        thresholds = json.load(f)
    return thresholds
