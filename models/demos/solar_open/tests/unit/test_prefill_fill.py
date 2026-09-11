# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host tests of tt/prefill_fill.py: the fill bound of a single-user eager prefill (phase 3g / L1)."""

import pytest

from models.demos.solar_open.tt.prefill_fill import fill_blocks_for_prompt, fill_table_columns


@pytest.mark.parametrize("last_token_idx", list(range(0, 1100, 7)) + [63, 64, 127, 128, 155, 191, 192, 1023])
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_fill_blocks_equals_ceil_of_real_tokens(last_token_idx, block_size):
    get_last_token = (last_token_idx // 32) * 32  # what the Generator passes
    assert fill_blocks_for_prompt(get_last_token, block_size) == -(-(last_token_idx + 1) // block_size)


def test_fill_blocks_none_without_last_token():
    assert fill_blocks_for_prompt(-1, 64) is None
    assert fill_blocks_for_prompt(None, 64) is None


@pytest.mark.parametrize(
    "prompt_len, padded, expected",
    [
        (156, 1024, 3),  # the live case: 3 real blocks of 16 columns
        (339, 1024, 6),
        (716, 1024, 12),
        (936, 1024, 15),
        (986, 1024, None),  # 16 real blocks: nothing to cut
        (1031, 2048, 17),
        (78, 128, None),  # 2 real blocks of 2 columns
        (40, 128, 1),  # a <= 64-token prompt: the second column is a pad entry
    ],
)
def test_exact_fit_eager_table_is_cut_to_the_real_blocks(prompt_len, padded, expected):
    glt = ((prompt_len - 1) // 32) * 32
    assert fill_table_columns(padded // 64, 64, padded, 1, glt, chunked=False) == expected


def test_traced_table_is_never_cut():
    # the traced path pads the table to the whole pool (Generator._pad_or_create_page_table(table, num_blocks_in_pool))
    assert fill_table_columns(10280, 64, 128, 1, 64, chunked=False) is None
    assert fill_table_columns(10280, 64, 1024, 1, 128, chunked=False) is None


def test_packed_and_chunked_and_demo_tables_are_left_alone():
    assert fill_table_columns(16, 64, 4096, 32, 128, chunked=False) is None  # packed pass
    assert fill_table_columns(16, 64, 1024, 1, 128, chunked=True) is None  # chunked prefill
    assert fill_table_columns(64, 64, 1024, 1, 128, chunked=False) is None  # a 64-block demo table (not exact-fit)
    assert fill_table_columns(16, 64, 1024, 1, -1, chunked=False) is None  # no last-token slicing
