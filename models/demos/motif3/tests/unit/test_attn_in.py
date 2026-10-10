# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase F F1 (``MOTIF3_ATTN_IN=fused``, ``tt/kernels/attn_in.py``): the fused decode attention input chain equals the
op chain bit for bit.

Device (lock wrapper)::

    S=/home/ttuser/hchang/experiments/motif-3/scripts
    $S/devrun.sh -t 900 -n attn_in -- python -m pytest models/demos/motif3/tests/unit/test_attn_in.py -s \
        -p no:cacheprovider

* ``test_attn_in_kernel_bitwise``: SWA layer 1 and global layer 0 (random weights), 8 rows (5 inactive lanes) and
  16 rows (the T64 step): q_mla, g, lam, kv_row (DRAM mode), the draft-1 update input (shard mode, 8 rows) and the
  debug cq_n equal the op chain on all 32 chips; 20 repeated calls are bitwise identical (determinism).
* ``test_attn_in_forward_decode``: ``forward_decode`` with ``attn_in`` "ops" vs "fused": output and written cache bitwise
  equal (layers 1 and 0, the draft-1 row write); traced cost of both modes.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.motif3.tt.attention import MotifAttention
from models.demos.motif3.tests.unit.test_attention import (
    MESH,
    _chip_index,
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


def ops_chain(attn, cq, kvl, cos, sin, update_mc=None):
    """The release's input chain after the latent projections (``MotifAttention._project`` / ``_input_chain_ops``
    without the cache write) -> dict of device tensors. Consumes ``kvl``."""
    cfg = attn.cfg
    cq_n = ttnn.rms_norm(cq, epsilon=cfg.rms_norm_eps, compute_kernel_config=attn.ckc_norm)
    q = attn._linear(cq_n, attn.w_q_b, ckc=attn.ckc_heads, pc=attn._pc("wq_b", True))
    g = attn._linear(cq_n, attn.w_gate, ckc=attn.ckc_heads, pc=attn._pc("gate", True), activation="sigmoid")
    n, kpe, lam = attn._split_kv(kvl)
    q_nope, q_pe = ttnn.experimental.nlp_create_q_heads_split(q, num_heads=attn.H, split_head_dim=attn.nope)
    q_lat = ttnn.matmul(q_nope, attn.w_uk, program_config=attn._pc("w_uk", True),
                        compute_kernel_config=attn.ckc_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q_pe_r = attn._rope(q_pe, cos, sin)
    q_heads = ttnn.concat([q_lat, q_pe_r], dim=-1)
    q_mla = ttnn.transpose(q_heads, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_pe = attn._rope(kpe, cos, sin)
    kv_row = ttnn.concat([n, k_pe], dim=-1)
    out = dict(cq_n=cq_n, q_mla=q_mla, g=g, lam=lam, kv_row=kv_row)
    if update_mc is not None:
        out["kv_upd"] = ttnn.transpose(kv_row, 1, 2, memory_config=update_mc)
    _free([q, n, kpe, q_nope, q_pe, q_lat, q_pe_r, q_heads, k_pe])
    return out


def _poison(mesh_device, rows: int) -> None:
    """Allocate and free 7.0-filled tensors of the fused outputs' shapes, so the next outputs (same sizes) most likely
    land on poisoned buffers: an output the kernel did not write cannot pass as a stale correct one."""
    ts = [ttnn.full(ttnn.Shape(s), 7.0, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
          for s, dt in (([1, rows, 10, 576], ttnn.bfloat16), ([1, 1, rows, 1024], ttnn.bfloat16),
                        ([1, 1, rows, 64], ttnn.bfloat16), ([1, 1, rows, 576], ttnn.bfloat16),
                        ([1, 1, rows, 1024], ttnn.float32))]
    _free(ts)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_attn_in_kernel_bitwise(mesh_device, device_params):
    from models.demos.motif3.tt.kernels.attn_in import FusedAttnIn

    cfg, ccl, rope = _setup(mesh_device, "attn_in kernel")
    args = ref_args()
    B, block, ctx = cfg.max_batch, cfg.kv_block_size, 1024
    Wd = ctx // block
    pool = 1 + B * Wd
    g = torch.Generator().manual_seed(11)
    pt = (torch.randperm(pool - 1, generator=g) + 1).to(torch.int32).reshape(B, Wd)
    pos = [ctx - 1 - 29 * i for i in range(B)]
    for i in (3, 11, 19, 27, 12):
        pos[i] = -1
    d = _decode_step_inputs(mesh_device, cfg, rope, pos, torch.randn(B, 4096, generator=g).bfloat16().float(), pt)
    fi = FusedAttnIn(mesh_device, eps=cfg.rms_norm_eps, debug=1)
    fi0 = FusedAttnIn(mesh_device, eps=cfg.rms_norm_eps, debug=0)
    fails = []
    for layer in (1, 0):
        attn = MotifAttention(mesh_device, cfg, layer, source=hf_source(random_attn_tensors(args, seed=70 + layer),
                                                                        layer), ccl=ccl, rope=rope, cache=False)
        cos, sin = attn._rot_tables(d["rot"])
        for rows in (8, 16):
            if rows == 8:
                x = d["x"]
            else:  # the T64 step's 16 rows per DP row
                from models.demos.motif3.tt.rope import shard_lanes

                xh = torch.randn(cfg.dp, 1, rows, 4096, generator=g).bfloat16().float()
                x = shard_lanes(xh, cfg, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=mesh_device)
            cq = attn._linear(x, attn.w_q_lat, ckc=attn.ckc_latent, pc=attn._pc("q_lat", True), dtype=ttnn.float32)
            kvl_ops = attn._kv_latent(x, True)
            kvl = attn._kv_latent(x, True)
            if ndiff(words(kvl_ops), words(kvl)):
                fails.append(f"L{layer} rows {rows}: the kv latent linear is not deterministic")
            umc = attn.update_mc if rows == 8 else None
            ref = ops_chain(attn, cq, kvl_ops, cos, sin, update_mc=umc)
            want = {k: words(v) for k, v in ref.items()}
            # DRAM kv_row mode (writers / T64); the outputs land on poisoned (7.0) buffers
            _poison(mesh_device, rows)
            qm, gg, lam, kvr, cqn = fi(cq, kvl, cos, sin, attn.w_q_b, attn.w_gate, attn.w_uk)
            got = dict(cq_n=words(cqn), q_mla=words(qm), g=words(gg), lam=words(lam), kv_row=words(kvr))
            keep = [qm, gg, lam, kvr, cqn]  # alive through the shard-mode call: its outputs get other buffers
            for k, v in got.items():
                nd = ndiff(want[k], v)
                log(f"attn_in L{layer} rows {rows} dram: {k} {'bitwise' if nd == 0 else f'{nd} words differ'}")
                if nd:
                    fails.append(f"L{layer} rows {rows} dram {k}: {nd} words differ")
            # stage "full": the latent projections in the same program, from x
            _poison(mesh_device, rows)
            o = fi.full(x, attn.w_q_lat, attn.w_kv_lat, cos, sin, attn.w_q_b, attn.w_gate, attn.w_uk, update_mc=umc)
            names = ("q_mla", "g", "lam", "kv_upd" if umc is not None else "kv_row", "cq_n")
            got = {k: words(v) for k, v in zip(names, o)}
            keep += list(o)
            for k, v in got.items():
                nd = ndiff(want[k], v)
                log(f"attn_in L{layer} rows {rows} full: {k} {'bitwise' if nd == 0 else f'{nd} words differ'}")
                if nd:
                    fails.append(f"L{layer} rows {rows} full {k}: {nd} words differ")
            if umc is not None:  # determinism of the full program: 20 repeated calls
                first, bad = None, 0
                for _ in range(20):
                    _poison(mesh_device, rows)
                    o = fi0.full(x, attn.w_q_lat, attn.w_kv_lat, cos, sin, attn.w_q_b, attn.w_gate, attn.w_uk,
                                 update_mc=umc)
                    w = [words(t) for t in o]
                    _free(list(o))
                    if first is None:
                        first = w
                    elif any(ndiff(a, b) for a, b in zip(first, w)):
                        bad += 1
                log(f"attn_in L{layer} full: 20 repeated calls, {bad} differ from the first")
                if bad:
                    fails.append(f"L{layer} full: {bad} / 19 repeated calls differ")
            if umc is not None:  # draft-1 shard mode
                _poison(mesh_device, rows)
                qm, gg, lam, kvu = fi0(cq, kvl, cos, sin, attn.w_q_b, attn.w_gate, attn.w_uk, update_mc=umc)
                got = dict(q_mla=words(qm), g=words(gg), lam=words(lam), kv_upd=words(kvu))
                _free([qm, gg, lam, kvu])
                for k, v in got.items():
                    nd = ndiff(want[k], v)
                    log(f"attn_in L{layer} rows {rows} shard: {k} {'bitwise' if nd == 0 else f'{nd} words differ'}")
                    if nd:
                        fails.append(f"L{layer} rows {rows} shard {k}: {nd} words differ")
                # determinism: 20 repeated calls
                first = None
                bad = 0
                for _ in range(20):
                    _poison(mesh_device, rows)
                    o = fi0(cq, kvl, cos, sin, attn.w_q_b, attn.w_gate, attn.w_uk, update_mc=umc)
                    w = [words(t) for t in o]
                    _free(list(o))
                    if first is None:
                        first = w
                    elif any(ndiff(a, b) for a, b in zip(first, w)):
                        bad += 1
                log(f"attn_in L{layer}: 20 repeated calls, {bad} differ from the first")
                if bad:
                    fails.append(f"L{layer}: {bad} / 19 repeated calls differ")
            _free(list(ref.values()) + [cq, kvl] + keep)
            if rows != 8:
                _free(x)
        del attn
    _free_step(d)
    assert not fails, "; ".join(fails)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_attn_in_forward_decode(mesh_device, device_params):
    cfg, ccl, rope = _setup(mesh_device, "attn_in forward")
    args = ref_args()
    B, block, ctx = cfg.max_batch, cfg.kv_block_size, 1024
    Wd = ctx // block
    pool = 1 + B * Wd
    g = torch.Generator().manual_seed(13)
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

    dp_chips = [_chip_index(cfg, r) for r in range(cfg.dp)]
    fails = []
    for layer in (1, 0):
        attn = MotifAttention(mesh_device, cfg, layer, source=hf_source(random_attn_tensors(args, seed=80 + layer),
                                                                        layer), ccl=ccl, rope=rope, cache=False)
        outs = {}
        for mode in ("ops", "post", "fused"):
            attn.attn_in = mode
            cache = rep(cache_h, cfg.dtypes.kv_cache)
            o = attn.forward_decode(d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache,
                                    active=d["act"])
            outs[mode] = (words(o), words(cache, dp_chips))
            _free([o, cache])
        for mode in ("post", "fused"):
            n_o = ndiff(outs["ops"][0], outs[mode][0])
            n_c = ndiff(outs["ops"][1], outs[mode][1])
            log(f"attn_in forward_decode L{layer} {mode}: output {n_o} words differ, cache {n_c} words differ")
            if n_o or n_c:
                fails.append(f"L{layer} {mode}: output {n_o} / cache {n_c} words differ")
        cache = rep(cache_h, cfg.dtypes.kv_cache)
        us = {}
        for mode in ("ops", "post", "fused"):
            attn.attn_in = mode
            us[mode] = _traced_us(mesh_device, lambda: attn.forward_decode(
                d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache, active=d["act"]),
                n=32, reps=5)["slope_us"]
        _free(cache)
        log(f"attn_in forward_decode L{layer}: traced ops {us['ops']:.1f} us, post {us['post']:.1f} us "
            f"({us['post'] - us['ops']:+.1f}), fused {us['fused']:.1f} us ({us['fused'] - us['ops']:+.1f})")
        del attn
    _free_step(d)
    assert not fails, "; ".join(fails)
