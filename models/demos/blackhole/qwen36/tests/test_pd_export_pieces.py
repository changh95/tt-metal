# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the piecewise KV export's planning and host assembly (pd_transfer.py, no device):

* `export_pieces`: every request is covered by pre-warmed power-of-two buckets no larger than the pool's reach, full
  pieces first, and the remainder's padding tax is traded against the per-piece fixed cost (129 -> 128 + 1, 255 -> 256);
* `_piece_row_map` + `_copy_rows`: the device-major rows of a padded piece (runs mode, incl. descending runs and
  top-of-pool padding; blocks mode) land block-major in the right output rows, in one pass, for a simulated read.

Run: python -m pytest models/demos/blackhole/qwen36/tests/test_pd_export_pieces.py -q
"""
import pytest
import torch

from models.demos.blackhole.qwen36.tt.pd_transfer import (
    _copy_rows,
    _piece_row_map,
    coalesce_runs,
    export_bucket,
    export_pieces,
    pad_block_ids,
)


def _check_plan(n, pieces, max_piece):
    assert sum(cnt for cnt, _ in pieces) == n
    for cnt, bucket in pieces:
        assert 1 <= cnt <= bucket <= max_piece
        assert bucket == export_bucket(bucket), f"{bucket} is not a warmed bucket"
        assert export_bucket(cnt) == bucket, f"{cnt} real blocks in a {bucket} bucket: over-padded"


@pytest.mark.parametrize(
    "n,expected",
    [
        (1, [(1, 1)]),
        (64, [(64, 64)]),
        (129, [(128, 128), (1, 1)]),  # 8k + template: no 256-bucket padding tax
        (192, [(128, 128), (64, 64)]),
        (200, [(200, 256)]),  # 56 pad blocks (~42 ms) vs a second piece (~37 ms): single bucket
        (255, [(255, 256)]),
        (256, [(256, 256)]),
        (257, [(256, 256), (1, 1)]),  # 16k
        (513, [(256, 256)] * 2 + [(1, 1)]),  # 32k
        (1025, [(256, 256)] * 4 + [(1, 1)]),  # 64k
        (2049, [(256, 256)] * 8 + [(1, 1)]),  # 128k
        (65, [(64, 64), (1, 1)]),
        (96, [(96, 128)]),
    ],
)
def test_export_pieces_default_reach(n, expected):
    pieces = export_pieces(n, 256, fixed=48)
    _check_plan(n, pieces, 256)
    assert pieces == expected


@pytest.mark.parametrize("max_piece", [64, 256, 1024])
@pytest.mark.parametrize("fixed", [0, 48, 10**6])
def test_export_pieces_cover_every_length(max_piece, fixed):
    for n in list(range(1, 600)) + [1023, 1024, 1025, 2047, 2048, 2049, 4097]:
        pieces = export_pieces(n, max_piece, fixed=fixed)
        _check_plan(n, pieces, max_piece)
        if fixed >= 10**6:  # a piece costs everything: the remainder is one padded bucket
            assert len(pieces) == n // max_piece + (1 if n % max_piece else 0)
        if fixed == 0:  # padding costs everything: exact set-bit pieces, no padding at all
            assert all(cnt == bucket for cnt, bucket in pieces)


def test_export_pieces_probe_reach_1024():
    """QWEN36_PD_EXPORT_POOL_MAX_BLOCKS=1024 (the zero-code probe): 2049 blocks = 1024 + 1024 + 1."""
    assert export_pieces(2049, 1024) == [(1024, 1024), (1024, 1024), (1, 1)]


def _simulate_device_read(ref, ids, runs, mode, n_dev):
    """What the device hands back for one padded piece as the pool sees it after `.view(n_dev, n_caches, bucket,
    nkv, blk, hd)`: run regions ascending in `runs` order (runs mode) or the padded ids in order (blocks mode),
    device-major. `ref` is the per-cache block-major reference [num_blocks, n_dev * nkv, blk, hd]."""
    if mode == "runs":
        order = [b for lo, hi, _ in runs for b in range(lo, hi)]
    else:
        order = list(ids)
    n_caches = len(ref)
    bucket = len(order)
    nb, ndn, blk, hd = ref[0].shape
    nkv = ndn // n_dev
    host = torch.empty(n_dev, n_caches, bucket, nkv, blk, hd, dtype=torch.bfloat16)
    for ci, cache in enumerate(ref):
        # [bucket, n_dev, nkv, blk, hd] -> device-major
        host[:, ci] = cache[order].view(bucket, n_dev, nkv, blk, hd).permute(1, 0, 2, 3, 4)
    return host


@pytest.mark.parametrize("n_dev", [1, 4])
@pytest.mark.parametrize(
    "name,ids,mode",
    [
        ("asc_contig", list(range(10, 15)), "runs"),
        ("desc_contig", list(range(14, 9, -1)), "runs"),
        ("top_of_pool_asc", list(range(61, 64)), "runs"),  # padded below: real rows follow the pad rows
        ("top_of_pool_desc", [63, 62, 61], "runs"),
        ("two_runs_forced_runs", [3, 4, 5, 9, 8], "runs"),
        ("fragmented", [38, 57, 50, 3, 4], "blocks"),
        ("one", [7], "runs"),
    ],
)
def test_piece_rows_land_block_major(n_dev, name, ids, mode):
    torch.manual_seed(0)
    num_blocks, nkv, blk, hd, n_caches = 64, 1, 4, 8, 3
    ref = [torch.randn(num_blocks, n_dev * nkv, blk, hd).to(torch.bfloat16) for _ in range(n_caches)]
    cnt = len(ids)
    bucket = export_bucket(cnt)
    padded, real_off = pad_block_ids(ids, bucket, num_blocks)
    assert padded[real_off : real_off + cnt] == ids
    runs = coalesce_runs(padded)
    if mode == "runs" and name != "two_runs_forced_runs":
        assert len(runs) == 1, "padding must keep a single run a single run"
    host = _simulate_device_read(ref, padded, runs, mode, n_dev)
    rows = _piece_row_map(runs, mode, real_off, cnt)
    assert len(rows) == cnt and len(set(rows)) == cnt and all(0 <= r < bucket for r in rows)
    # the piece is written at offset `off` of a larger request's output
    off, n_total = 3, 3 + cnt + 2
    for ci in range(n_caches):
        out = torch.zeros(n_total, n_dev * nkv, blk, hd, dtype=torch.bfloat16)
        src = host[:, ci].permute(1, 0, 2, 3, 4)
        dst = out[off : off + cnt].view(cnt, n_dev, nkv, blk, hd)
        _copy_rows(dst, src, rows)
        assert torch.equal(out[off : off + cnt], ref[ci][ids]), f"{name} cache {ci}"
        assert torch.count_nonzero(out[:off]) == 0 and torch.count_nonzero(out[off + cnt :]) == 0
        assert out.is_contiguous()


def test_copy_rows_is_in_place_and_single_pass():
    src = torch.arange(6 * 2 * 3, dtype=torch.float32).view(6, 2, 3).permute(0, 2, 1)  # non-contiguous view
    dst = torch.zeros(4, 3, 2)
    _copy_rows(dst, src, [1, 2, 3, 4])  # contiguous ascending: strided copy
    assert torch.equal(dst, src[1:5])
    dst2 = torch.zeros(3, 3, 2)
    ptr = dst2.data_ptr()
    _copy_rows(dst2, src, [5, 0, 2])  # gather straight into dst
    assert dst2.data_ptr() == ptr and torch.equal(dst2, src[[5, 0, 2]])
