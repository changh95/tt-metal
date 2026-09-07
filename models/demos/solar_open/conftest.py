# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures shared by every Solar-Open test and demo (unit tests, integration tests, text demo)."""

import json
import os
from pathlib import Path

import pytest

# PCC thresholds per component, keyed by ModelArgs.model_name ("Solar-Open-100B") and mode (decode/prefill).
# Resolved relative to this file so the fixture works from any working directory.
UNIT_TEST_THRESHOLDS_PATH = Path(__file__).resolve().parent / "unit_test_thresholds.json"


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
def test_thresholds(request):
    """Component PCC thresholds: ``thresholds["Solar-Open-100B"][mode][component]``."""
    with open(UNIT_TEST_THRESHOLDS_PATH, "r") as f:
        thresholds = json.load(f)
    return thresholds
