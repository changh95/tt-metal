# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side (pure torch) value builders for the traced masked-bucket prefill.

The eager masked-bucket prefill rebuilds ~151 small host tensors per request (GDN beta/g and
q/k/v validity masks in every GDN layer, the FIR decode-window one-hot in every GDN layer, the
KV-fill page table, the logits one-hot). Each of those is a ``ttnn.from_torch`` host write, which
is illegal inside a captured trace ("Writes are not supported during trace capture") and is also
the bulk of the per-request host dispatch cost.

The traced path hoists all of them into ONE model-owned set of persistent device buffers per
bucket whose SHAPES depend only on the bucket, so a single captured program set serves every
``actual_len``; only the VALUES change per request and are DMA'd in with
``ttnn.copy_host_to_device_tensor`` before ``ttnn.execute_trace``.

Everything here is pure torch (no ttnn import) so the value logic — the part that is easy to get
subtly wrong — is unit-testable on a host without a Tenstorrent device.
"""

import torch

# Default paged-KV block size for the Qwen3.6 serving path (get_block_size(kv_caches)).
DEFAULT_BLOCK_SIZE = 64


def host_masks(actual_len, bucket):
    """GDN validity mask [1, bucket, 1] float32: 1.0 for t < actual_len else 0.0.

    One tensor feeds BOTH device buffers: the fp32 beta/g mask and the bf16 q/k/v mask (the eager
    path in gdn/fused_chunk.py builds the same float32 ``_mt`` and uploads it twice, once as
    float32 and once as ``q.dtype``). 0.0/1.0 are exact in bf16, so the traced values are
    bit-identical to the eager ones.
    """
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    m = torch.zeros(1, bucket, 1, dtype=torch.float32)
    m[:, :actual_len, :] = 1.0
    return m


def host_conv_sel(actual_len, bucket, kernel_size):
    """FIR decode-window one-hot [1, K-1, bucket+K-1] float32 (sel[0, j, actual_len+j] = 1.0).

    Selects rows ``x_padded[:, actual_len : actual_len+(K-1)]`` — i.e. the last K-1 REAL conv
    inputs ``x[actual_len-(K-1) : actual_len]`` — via a matmul instead of a static slice, so the
    program depends only on shapes (fixed per bucket) and only the values depend on actual_len.
    Mirrors the one-hot built inside ``_causal_conv1d_fir`` for the eager int-valid_len path.
    """
    assert kernel_size >= 2, f"kernel_size {kernel_size} must be >= 2"
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    total_len = (kernel_size - 1) + bucket
    sel = torch.zeros(1, kernel_size - 1, total_len, dtype=torch.float32)
    for j in range(kernel_size - 1):
        sel[:, j, actual_len + j] = 1.0
    return sel


def host_conv_sel_split(actual_len, bucket, kernel_size):
    """host_conv_sel split for the fused KDA conv path: (sel_x [1, K-1, bucket], sel_c [1, K-1, K-1]) float32.

    The FIR one-hot indexes x_padded = concat(carry [K-1 rows], x [bucket rows]); column c of it is carry row c
    for c < K-1 and x row c-(K-1) otherwise. Splitting it by that boundary gives two one-hots such that
    ``sel_x @ x + sel_c @ carry == host_conv_sel @ concat(carry, x)`` exactly (each output row has a single 1.0 in
    exactly one of the two), so the fused path never has to materialize the concat. sel_c is non-zero only when
    actual_len < K-1 (the window still reaches into the previous chunk's carry).
    """
    sel = host_conv_sel(actual_len, bucket, kernel_size)
    sel_c = sel[:, :, : kernel_size - 1].clone().contiguous()
    sel_x = sel[:, :, kernel_size - 1 :].clone().contiguous()
    return sel_x, sel_c


def host_logit_sel(actual_len, bucket):
    """Last-real-row one-hot [1, 1, 1, bucket] float32 for the TP logits select (row actual_len-1)."""
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    sel = torch.zeros(1, 1, 1, bucket, dtype=torch.float32)
    sel[0, 0, 0, actual_len - 1] = 1.0
    return sel


def fill_pt_row(page_table, chunk_start, actual_len, bucket, pad_block, block_size=DEFAULT_BLOCK_SIZE):
    """FIXED-WIDTH KV-fill page table [1, bucket//block_size] int32 for the traced bucket body.

    The eager path sizes this table to the REAL blocks only (``page_table[:, blk0:blkN]`` with
    blkN = ceil((chunk_start+actual_len)/block_size)), so its width — and hence the paged_fill_cache
    program plus the ``ttnn.slice`` that trims K/V to page_len — changes with actual_len. A trace
    needs one fixed shape, so the traced body always fills the whole bucket: width = bucket/block_size
    makes ``page_len == S``, which also removes the slice entirely.

    Entries ``[0, nreal)`` are the request's real blocks. Trailing entries hold K/V for PAD rows,
    which causal SDPA never reads (k <= q < actual_len) and decode later overwrites via
    paged_update_cache; they must nevertheless point somewhere harmless:
      * the request's OWN mapped block at that index when it is non-zero (the natural full-bucket
        fill — no extra block needed, and the demo/tests' arange page tables always hit this), else
      * ``pad_block``, a scratch physical block no request owns (default: the last KV block, env
        QWEN36_PREFILL_BUCKET_PAD_BLOCK). Never 0, because block 0 is a real request's block for a
        zero-padded page-table row.
    """
    assert bucket % block_size == 0, f"bucket {bucket} must be a multiple of block_size {block_size}"
    assert chunk_start % block_size == 0, f"chunk_start {chunk_start} must be block-aligned"
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    assert int(pad_block) != 0, "pad_block must not be block 0 (it aliases a real request's block)"
    width = bucket // block_size
    blk0 = chunk_start // block_size
    nreal = -(-(chunk_start + actual_len) // block_size) - blk0  # ceil div
    nreal = max(0, min(nreal, width))
    pt = page_table.reshape(1, -1)
    assert blk0 + nreal <= pt.shape[1], (
        f"page table row of {pt.shape[1]} blocks does not cover the {nreal} real block(s) at "
        f"offset {blk0} for chunk_start={chunk_start}, actual_len={actual_len}"
    )
    row = torch.full((1, width), int(pad_block), dtype=torch.int32)
    for j in range(width):
        idx = blk0 + j
        if idx >= pt.shape[1]:
            continue  # page-table row ends before the bucket -> scratch block
        v = int(pt[0, idx])
        if j < nreal or v != 0:
            row[0, j] = v
    return row


def parse_bucket_trace_gate(value, all_buckets):
    """Parse QWEN36_PREFILL_BUCKET_TRACE into a sorted tuple of buckets to trace.

    "0"/""/unset -> () (today's eager masked path, byte-for-byte). "1" -> every bucket.
    A comma list ("128", "128,256") -> exactly those buckets, so the 128 bucket can ship alone
    while the trace-region footprint of the larger ones is measured.
    """
    if value is None:
        return ()
    value = value.strip()
    if value in ("", "0", "off", "false"):
        return ()
    if value in ("1", "all", "true"):
        return tuple(sorted(all_buckets))
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        b = int(part)
        assert b in all_buckets, f"QWEN36_PREFILL_BUCKET_TRACE bucket {b} is not one of {tuple(all_buckets)}"
        out.append(b)
    return tuple(sorted(set(out)))


# --------------------------------------------------------------------------------------------- #
# Grouped traced prefill: B same-bucket users per trace (model.py capture_prefill_group_traces)
# --------------------------------------------------------------------------------------------- #
# The per-user traced bucket above prefills the N users of one serving step one after another
# (~100 ms of fixed cost each at 128 tokens). The grouped traces run B in {2, 4, 8} users of the
# same bucket through ONE body whose rows are user-major (row u*bucket + t = user u, position t);
# every per-user input becomes one row of a [B, ...] persistent buffer. Rows past the number of
# real users are DUMMY rows (a 1-token prompt over the scratch KV block) whose outputs are dropped.

# Grouped bodies run every prefill program at M = B * bucket rows; MAX_GROUP_ROWS is the chunk size, the largest M
# any prefill program (matmul/CCL/MLP CB budgets) is validated at -- 512 x 8 = 4096 rows overflowed the MLP's L1
# circular buffers at eager compile time, which no TRACE-region check can catch, so the cap is enforced at parse time.
MAX_GROUP_ROWS = 2048
DEFAULT_GROUP_TRACE_SPEC = "128:2,4,8;256:2,4,8;512:2,4"
DUMMY_ROW_LEN = 1


def _group_rows(actual_lens, B):
    """Per-row real lengths for a B-row body: the real users first, then DUMMY_ROW_LEN for pad rows."""
    assert 1 <= len(actual_lens) <= B, f"{len(actual_lens)} users do not fit a {B}-row group"
    return [int(a) for a in actual_lens] + [DUMMY_ROW_LEN] * (B - len(actual_lens))


def host_group_masks(actual_lens, bucket, B):
    """GDN validity mask [B, bucket, 1] float32, row u = host_masks(actual_lens[u]); dummy rows keep one valid
    token so every row of the batched chunk recurrence sees a well-formed prompt."""
    rows = _group_rows(actual_lens, B)
    return torch.cat([host_masks(a, bucket) for a in rows], dim=0)


def host_group_conv_sel_x(actual_lens, bucket, kernel_size, B):
    """Per-row fused-KDA decode-window one-hots over x: list of B [1, K-1, bucket] float32 (the `sel_x` half of
    host_conv_sel_split; the carry half is all zeros for a from-scratch prefill and is not used)."""
    return [host_conv_sel_split(a, bucket, kernel_size)[0] for a in _group_rows(actual_lens, B)]


def host_group_tokens(token_rows, bucket, B):
    """Right-padded token matrix [B, bucket] int32: row u = token_rows[u] (a [1, T_u] or [T_u] tensor, T_u <= bucket);
    pad and dummy rows are token 0."""
    assert 1 <= len(token_rows) <= B
    out = torch.zeros(B, bucket, dtype=torch.int32)
    for u, row in enumerate(token_rows):
        row = row.reshape(-1).to(torch.int32)
        assert 1 <= row.shape[0] <= bucket, f"row {u}: {row.shape[0]} tokens do not fit bucket {bucket}"
        out[u, : row.shape[0]] = row
    return out


def host_group_logit_sel(actual_lens, bucket, B):
    """Last-real-row one-hot [1, 1, B, B*bucket] float32 over the user-major hidden rows: sel[0, 0, u,
    u*bucket + actual_lens[u] - 1] = 1 for each real user (dummy rows select their row 0; discarded)."""
    rows = _group_rows(actual_lens, B)
    sel = torch.zeros(1, 1, B, B * bucket, dtype=torch.float32)
    for u, a in enumerate(rows):
        assert 1 <= a <= bucket
        sel[0, 0, u, u * bucket + a - 1] = 1.0
    return sel


def host_group_fill_pt(pt_rows, actual_lens, bucket, B, pad_block, block_size=DEFAULT_BLOCK_SIZE):
    """KV-fill page table [B, bucket//block_size] int32 for a B-row body (paged_fill_cache reads row u for
    batch_idx=u): real rows via fill_pt_row over the request's page-table row; dummy rows fill only the scratch
    block (whose contents never reach a live request)."""
    assert len(pt_rows) == len(actual_lens) and 1 <= len(pt_rows) <= B
    width = bucket // block_size
    out = torch.full((B, width), int(pad_block), dtype=torch.int32)
    for u, (pt, a) in enumerate(zip(pt_rows, actual_lens)):
        out[u : u + 1] = fill_pt_row(pt, 0, int(a), bucket, pad_block, block_size)
    return out


def parse_group_trace_spec(value, all_buckets, default=DEFAULT_GROUP_TRACE_SPEC):
    """Parse QWEN36_PREFILL_GROUP_TRACE into {bucket: (B, ...)} (sorted, B in {2, 4, 8}).

    unset/"0"/""/off -> {} (per-user traces only: the grouped traces are OPT-IN); "1"/all/default -> the default
    spec; otherwise a spec of the form "128:2,4,8;256:4,8" (bucket:comma-separated group sizes, ';'-separated)."""
    if value is None:
        return {}
    value = value.strip()
    if value in ("1", "all", "true", "default"):
        value = default
    if value in ("", "0", "off", "false"):
        return {}
    out = {}
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        bucket_s, _, bs = entry.partition(":")
        bucket = int(bucket_s)
        assert bucket in all_buckets, f"QWEN36_PREFILL_GROUP_TRACE bucket {bucket} is not one of {tuple(all_buckets)}"
        sizes = tuple(sorted({int(b) for b in bs.split(",") if b.strip()}))
        assert sizes and all(b in (2, 4, 8) for b in sizes), f"group sizes {sizes} must be in (2, 4, 8)"
        for b in sizes:
            assert b * bucket <= MAX_GROUP_ROWS, (
                f"QWEN36_PREFILL_GROUP_TRACE {bucket}:{b} = {b * bucket} rows exceeds MAX_GROUP_ROWS={MAX_GROUP_ROWS} "
                f"(the chunk size; prefill programs are not validated at larger M)"
            )
        out[bucket] = sizes
    return out


def plan_prefill_groups(buckets, available):
    """Split one prefill step's users into grouped traces and per-user leftovers.

    buckets:   per-user masked bucket (model._mask_bucket_for(actual_len)), in call order.
    available: {bucket: sorted tuple of captured group sizes B}.
    Returns (groups, singles): groups = [(bucket, B, [user indices...]), ...] with 2 <= len(users) <= B (the
    smallest captured B that fits; rows past len(users) are dummies), singles = user indices that run the
    per-user path (buckets without group traces, and a lone leftover user per bucket)."""
    groups, singles = [], []
    by_bucket = {}
    for u, b in enumerate(buckets):
        if available.get(b):
            by_bucket.setdefault(b, []).append(u)
        else:
            singles.append(u)
    for b in sorted(by_bucket):
        us = by_bucket[b]
        sizes = sorted(available[b])
        while len(us) >= 2:
            take, us = us[: sizes[-1]], us[sizes[-1] :]
            B = next(s for s in sizes if s >= len(take))
            groups.append((b, B, take))
        singles.extend(us)
    return groups, sorted(singles)
