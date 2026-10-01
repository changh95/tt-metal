# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The reference must stay device-free: importing it never imports ttnn, and this test session has not either
(fails loudly if the suite is run with tt-metal's root conftest, which imports ttnn)."""

import subprocess
import sys


def test_reference_package_does_not_import_ttnn():
    code = (
        "import sys\n"
        "import models.demos.motif3.reference as r\n"
        "import models.demos.motif3.reference.generate, models.demos.motif3.reference.golden\n"
        "import models.demos.motif3.reference.tokenizer, models.demos.motif3.reference.weights\n"
        "bad = sorted(m for m in sys.modules if m == 'ttnn' or m.startswith('ttnn.') or m.startswith('tt_lib'))\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0 and res.stdout.strip().endswith("ok"), res.stderr[-2000:]


RUN_HINT = (
    'run with: pytest -p no:cacheprovider --noconftest -o addopts="" --import-mode=importlib '
    "(tt-metal's conftest imports ttnn; see README 'Running the tests')"
)


def test_session_has_not_loaded_ttnn():
    assert "ttnn" not in sys.modules, RUN_HINT
