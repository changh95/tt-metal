# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM parser plugins for Motif-3: reasoning parser ``motif`` and tool-call parser ``motif`` / ``motif_hermes``.

Upstream vLLM 0.26.0 ships neither (design 00 §7.2 Q7, §7.3 decision 6; study 01 §7.4; study 05 §12). The two files
next to this one are Motif Technologies' own parsers from https://github.com/MotifTechnologies/vllm @ 4cd9eb4
(Apache-2.0), unchanged except for an attribution header, packaged as drop-in plugin files::

    vllm serve <Motif-3 snapshot> --trust-remote-code ... \\
      --reasoning-parser-plugin <this dir>/motif_reasoning_parser.py --reasoning-parser motif \\
      --tool-parser-plugin <this dir>/motif_tool_parser.py --tool-call-parser motif --enable-auto-tool-choice

* ``motif`` reasoning: ``DeepSeekR1ReasoningParser`` (``<think>`` ... ``</think>``; Motif's generation prompt already
  opens ``<think>``) that hands everything to ``IdentityReasoningParser`` when the request sets
  ``chat_template_kwargs={"enable_thinking": false}``.
* ``motif`` tool calls: ``Hermes2ProToolParser`` (``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``) with
  Motif's deterministic repair ladder for malformed JSON inside each block.

``<think>`` / ``</think>`` / ``<tool_call>`` / ``</tool_call>`` are single, non-special tokens 11 / 12 / 13 / 14 of
the Motif tokenizer, so the parsers see them in the detokenized text (study 01 §7.1).

vLLM imports these files by *path* (``import_from_path``) in the API server, and the reasoning parser also in
EngineCore for structured outputs, so they import only vLLM. This ``__init__`` imports nothing at all.
"""

from pathlib import Path

_HERE = Path(__file__).resolve().parent

REASONING_PARSER_PLUGIN = str(_HERE / "motif_reasoning_parser.py")
TOOL_PARSER_PLUGIN = str(_HERE / "motif_tool_parser.py")
REASONING_PARSER_NAME = "motif"
TOOL_PARSER_NAME = "motif"


def vllm_cli_args(reasoning: bool = True, tools: bool = True) -> list:
    """The ``vllm serve`` flags that load and select the Motif parsers."""
    args = []
    if reasoning:
        args += ["--reasoning-parser-plugin", REASONING_PARSER_PLUGIN, "--reasoning-parser", REASONING_PARSER_NAME]
    if tools:
        args += [
            "--tool-parser-plugin",
            TOOL_PARSER_PLUGIN,
            "--tool-call-parser",
            TOOL_PARSER_NAME,
            "--enable-auto-tool-choice",
        ]
    return args


__all__ = [
    "REASONING_PARSER_NAME",
    "REASONING_PARSER_PLUGIN",
    "TOOL_PARSER_NAME",
    "TOOL_PARSER_PLUGIN",
    "vllm_cli_args",
]
