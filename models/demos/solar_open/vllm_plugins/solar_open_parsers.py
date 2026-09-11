# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
vLLM parser plugin for Solar-Open-100B: importing this file registers Upstage's ``SolarOpenToolParser`` and, as
``SolarOpenTTReasoningParser``, Upstage's ``SolarOpenReasoningParser`` with the streaming / prefill fixes of
``vllm_support.SolarOpenReasoningParserFixes`` (both parsers ship in the HF repo,
``$HF_MODEL/solar_open_{reasoning,tool}_parser.py``, without a register decorator) under the name ``solar_open``;
Upstage's reasoning parser as shipped stays reachable as ``solar_open_upstream`` (``--reasoning-parser
solar_open_upstream`` reproduces the raw-marker streaming of reasoning_effort low, see ``vllm_support``).

    vllm serve upstage/Solar-Open-100B ... \\
        --reasoning-parser-plugin models/demos/solar_open/vllm_plugins/solar_open_parsers.py --reasoning-parser solar_open \\
        --tool-parser-plugin models/demos/solar_open/vllm_plugins/solar_open_parsers.py --tool-call-parser solar_open \\
        --enable-auto-tool-choice

vLLM imports a plugin file by path (``import_from_path``; both flags may name this same file, it then runs twice and
re-registers with ``force=True``), so the tt-metal root must be on PYTHONPATH (env.sh; tt-inference-server exports it)
and HF_MODEL must point at the snapshot directory (the server's ``model_file_symlinks_map/Solar-Open-100B`` symlink).
``register_vllm_parsers`` first installs the vLLM-0.12 import shims Upstage's files need on vLLM 0.25.1. Deliberately
a separate file: registering imports Upstage's parser module, which patches ``json._default_encoder`` at import, so
``tt/vllm_support.py`` never does it implicitly. Host-verified on vLLM 0.25.1 through
``ReasoningParserManager.import_reasoning_parser`` / ``ToolParserManager.import_tool_parser``
(``tests/unit/test_vllm_wrapper_import.py``) and live on 2026-09-10 (reasoning / content split, streamed reasoning
deltas, a tool call with a non-empty id; README "Serving with vLLM and tt-inference-server").
"""

from models.demos.solar_open.tt.vllm_support import register_vllm_parsers

register_vllm_parsers()
