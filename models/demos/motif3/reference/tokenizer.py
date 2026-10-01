# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 tokenizer + chat template without ``trust_remote_code`` (offline, transformers 5.x).

The checkpoint ships ``tokenizer.json`` (SuperBPE, 220160 tokens, no automatic BOS) and ``chat_template.jinja``
(which starts every conversation with ``<|beginoftext|>`` and opens ``<think>`` in the generation prompt).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

from .weights import DEFAULT_WEIGHTS_DIR


def load_tokenizer(weights_dir: Optional[os.PathLike] = None):
    from transformers import PreTrainedTokenizerFast

    d = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    tok = PreTrainedTokenizerFast(tokenizer_file=str(d / "tokenizer.json"))
    tok.chat_template = (d / "chat_template.jinja").read_text()
    return tok


def encode_chat(
    messages: Sequence[dict],
    tokenizer=None,
    add_generation_prompt: bool = True,
    enable_thinking: Optional[bool] = None,
) -> List[int]:
    """Token ids of a chat (``[{"role": ..., "content": ...}, ...]``) rendered with the Motif-3 template."""
    tok = tokenizer or load_tokenizer()
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    enc = tok.apply_chat_template(list(messages), add_generation_prompt=add_generation_prompt, tokenize=True, **kwargs)
    ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc
    return [int(i) for i in ids]
