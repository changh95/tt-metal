# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase F F2 (``MOTIF3_ATTN_OUT``, ``tt/kernels/attn_out.py``): the fused decode attention output chain equals the
op chain bit for bit.

Device (lock wrapper)::

    S=/home/ttuser/hchang/experiments/motif-3/scripts
    $S/devrun.sh -t 900 -n attn_out -- python -m pytest models/demos/motif3/tests/unit/test_attn_out.py -s \
        -p no:cacheprovider

* ``test_attn_out_kernel_bitwise``: SWA layer 1 and global layer 0 (random weights), 8 rows (5 inactive lanes) and
  16 rows (the T64 step), random o_lat / g / lam: u, v = sigmoid(lam @ E), the wo input and the wo output of stages
  "uv" and "fused" equal the op chain (transpose, W_UV bmm, linear + sigmoid, D1 combine, wo linear) on all 32 chips;
  20 repeated "fused" calls are bitwise identical (determinism).
* ``test_attn_out_forward_decode``: ``forward_decode`` with ``attn_out`` "ops" / "uv" / "fused": output bitwise equal
  (layers 1 and 0, the draft-1 row write); traced cost of the three modes.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.motif3.tt.attention import MotifAttention
from models.demos.motif3.tests.unit.test_attention import (
    MESH,
    _decode_step_inputs,
    _free,
    _free_step,
    _setup,
    _traced_us,
    hf_source,
    log,
    random_attn_tensors,
    ref_args,
)


def words(t, idx=None):
    ts = ttnn.get_device_tensors(t)
    return [ttnn.to_torch(ts[i]).float().view(torch.int32) for i in (idx if idx is not None else range(len(ts)))]


def ndiff(a, b) -> int:
    return sum(int((p != q).sum()) for p, q in zip(a, b))


def ops_chain(attn, o_lat, g, lam, active):
    """The Phase E decode output chain before the AR(tp) (``forward_decode`` / ``_absorbed_epilogue`` with the fused
    D1 combine) -> dict of device tensors. Consumes nothing."""
    o_heads = ttnn.transpose(o_lat, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    u = ttnn.matmul(o_heads, attn.w_uv, program_config=attn._pc("w_uv", True), compute_kernel_config=attn.ckc_heads,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v = attn._linear(lam, attn.lam_expand, ckc=attn.ckc_heads, activation="sigmoid")
    dg = attn.fused_combine()(u, v, g, active)
    part = attn._linear(dg, attn.w_o, ckc=attn.ckc_heads, pc=attn._pc("wo", True))
    _free([o_heads])
    return dict(u=u, v=v, dg=dg, part=part)


def _inputs(mesh_device, cfg, rows, g):
    """Random o_lat [1, L, 10, 512], g = sigmoid(.) [1, 1, L, 1024], lam [1, 1, L, 64] and a 0/1 active mask with
    inactive lanes, one draw per DP row."""
    from models.demos.motif3.tt.rope import shard_lanes

    dp = cfg.dp

    def up(t):
        return shard_lanes(t.bfloat16().float(), cfg, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=mesh_device)

    o_lat = up(torch.randn(dp, rows, 10, 512, generator=g))
    gg = up(torch.sigmoid(torch.randn(dp, 1, rows, 1024, generator=g)))
    lam = up(2.0 * torch.randn(dp, 1, rows, 64, generator=g))
    act = torch.ones(dp, 1, rows, 1024)
    for r in range(dp):
        for lane in ((3 + r) % rows, (5 + 2 * r) % rows):
            act[r, 0, lane] = 0.0
    return o_lat, gg, lam, up(act)


def _poison(mesh_device, rows: int) -> None:
    """Allocate and free 7.0-filled tensors of the fused outputs' shapes, so the next outputs (same sizes) most likely
    land on poisoned buffers: an output the kernel did not write cannot pass as a stale correct one."""
    ts = [ttnn.full(ttnn.Shape(s), 7.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
          for s in ([1, 1, rows, 4096], [1, 1, rows, 1024], [1, 10, rows, 128])]
    _free(ts)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_attn_out_kernel_bitwise(mesh_device, device_params):
    from models.demos.motif3.tt.kernels.attn_out import FusedAttnOut

    cfg, ccl, rope = _setup(mesh_device, "attn_out kernel")
    args = ref_args()
    g = torch.Generator().manual_seed(21)
    fd = FusedAttnOut(mesh_device, debug=1)
    f0 = FusedAttnOut(mesh_device, debug=0)
    fails = []
    for layer in (1, 0):
        attn = MotifAttention(mesh_device, cfg, layer, source=hf_source(random_attn_tensors(args, seed=90 + layer),
                                                                        layer), ccl=ccl, rope=rope, cache=False)
        for rows in (8, 16):
            o_lat, gg, lam, act = _inputs(mesh_device, cfg, rows, g)
            ref = ops_chain(attn, o_lat, gg, lam, act)
            want = {k: words(v) for k, v in ref.items()}
            keep = []
            for stage in ("uv", "fused"):
                w_o = attn.w_o if stage == "fused" else None
                _poison(mesh_device, rows)
                out, du, dv, dd = fd(stage, o_lat, gg, lam, act, attn.w_uv, attn.lam_expand, w_o)
                got = dict(u=words(du), v=words(dv), dg=words(dd))
                if stage == "fused":
                    got["part"] = words(out)
                keep += [t for t in {id(x): x for x in (out, du, dv, dd)}.values()]
                for k, v in got.items():
                    nd = ndiff(want[k], v)
                    log(f"attn_out L{layer} rows {rows} {stage}: {k} {'bitwise' if nd == 0 else f'{nd} words differ'}")
                    if nd:
                        fails.append(f"L{layer} rows {rows} {stage} {k}: {nd} words differ")
                # production (debug 0) build of the same stage
                _poison(mesh_device, rows)
                o = f0(stage, o_lat, gg, lam, act, attn.w_uv, attn.lam_expand, w_o)
                nd = ndiff(want["part" if stage == "fused" else "dg"], words(o))
                log(f"attn_out L{layer} rows {rows} {stage} (debug 0): {'bitwise' if nd == 0 else f'{nd} words differ'}")
                if nd:
                    fails.append(f"L{layer} rows {rows} {stage} debug 0: {nd} words differ")
                _free([o])
            # determinism: 20 repeated fused calls
            first, bad = None, 0
            for _ in range(20):
                _poison(mesh_device, rows)
                o = f0("fused", o_lat, gg, lam, act, attn.w_uv, attn.lam_expand, attn.w_o)
                w = words(o)
                _free([o])
                if first is None:
                    first = w
                elif ndiff(first, w):
                    bad += 1
            log(f"attn_out L{layer} rows {rows} fused: 20 repeated calls, {bad} differ from the first")
            if bad:
                fails.append(f"L{layer} rows {rows} fused: {bad} / 19 repeated calls differ")
            _free(list(ref.values()) + keep + [o_lat, gg, lam, act])
        del attn
    assert not fails, "; ".join(fails)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_attn_out_forward_decode(mesh_device, device_params):
    cfg, ccl, rope = _setup(mesh_device, "attn_out forward")
    args = ref_args()
    B, block, ctx = cfg.max_batch, cfg.kv_block_size, 1024
    Wd = ctx // block
    pool = 1 + B * Wd
    g = torch.Generator().manual_seed(23)
    cache_h = torch.randn(pool, 1, block, cfg.kv_latent_dim, generator=g)
    pt = (torch.randperm(pool - 1, generator=g) + 1).to(torch.int32).reshape(B, Wd)
    pos = [ctx - 1 - 29 * i for i in range(B)]
    for i in (3, 11, 19, 27, 12):
        pos[i] = -1
    d = _decode_step_inputs(mesh_device, cfg, rope, pos, torch.randn(B, 4096, generator=g).bfloat16().float(), pt)

    def rep(t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    fails = []
    modes = ("ops", "uv", "fused")
    for layer in (1, 0):
        attn = MotifAttention(mesh_device, cfg, layer, source=hf_source(random_attn_tensors(args, seed=95 + layer),
                                                                        layer), ccl=ccl, rope=rope, cache=False)
        outs = {}
        for mode in modes:
            attn.attn_out = mode
            cache = rep(cache_h, cfg.dtypes.kv_cache)
            o = attn.forward_decode(d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache,
                                    active=d["act"])
            outs[mode] = words(o)
            _free([o, cache])
        for mode in modes[1:]:
            n_o = ndiff(outs["ops"], outs[mode])
            log(f"attn_out forward_decode L{layer} {mode}: output {n_o} words differ")
            if n_o:
                fails.append(f"L{layer} {mode}: output {n_o} words differ")
        cache = rep(cache_h, cfg.dtypes.kv_cache)
        us = {}
        for mode in modes:
            attn.attn_out = mode
            us[mode] = _traced_us(mesh_device, lambda: attn.forward_decode(
                d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache, active=d["act"]),
                n=32, reps=5)["slope_us"]
        _free(cache)
        log(f"attn_out forward_decode L{layer}: traced ops {us['ops']:.1f} us, uv {us['uv']:.1f} us "
            f"({us['uv'] - us['ops']:+.1f}), fused {us['fused']:.1f} us ({us['fused'] - us['ops']:+.1f})")
        del attn
    _free_step(d)
    assert not fails, "; ".join(fails)
