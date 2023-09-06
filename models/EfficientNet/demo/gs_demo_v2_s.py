"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

from models.EfficientNet.demo.demo_utils import run_gs_demo
from models.EfficientNet.tt.efficientnet_model import efficientnet_v2_s


def test_gs_demo_v2_s():
    run_gs_demo(efficientnet_v2_s)
