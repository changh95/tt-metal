# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
vLLM parser plugin for Solar-Open-100B: importing this file registers Upstage's ``SolarOpenReasoningParser`` and
``SolarOpenToolParser`` (shipped in the HF repo, ``$HF_MODEL/solar_open_{reasoning,tool}_parser.py``, without a
register decorator) under the name ``solar_open``.

    vllm serve upstage/Solar-Open-100B ... \\
        --reasoning-parser-plugin models/demos/solar_open/vllm_plugins/solar_open_parsers.py --reasoning-parser solar_open \\
        --tool-parser-plugin models/demos/solar_open/vllm_plugins/solar_open_parsers.py --tool-call-parser solar_open \\
        --enable-auto-tool-choice

vLLM imports a plugin file by path (``import_from_path``), so the tt-metal root must be on PYTHONPATH (env.sh) and
HF_MODEL must point at the snapshot directory. Deliberately a separate file: registering imports Upstage's parser
module, which patches ``json._default_encoder`` at import, so ``tt/vllm_support.py`` never does it implicitly.
UNTESTED against a live vLLM (not installed on the bring-up box).
"""

from models.demos.solar_open.tt.vllm_support import register_vllm_parsers

register_vllm_parsers()
