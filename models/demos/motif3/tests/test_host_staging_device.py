# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B6a device gate (docs/OPT_PHASE_A_REVIEW.md §7.1-7.2; logs/opt/phaseB/B6a): the host-only decode knobs
``MOTIF3_HOST_STAGING`` (``release`` | ``fast``) and ``MOTIF3_HOST_WAIT`` (``block`` | ``spin``) leave every decode
output bitwise unchanged.

One ``MotifGenerator`` (real weights, layers 0..3, device sampler on, the plain ``row`` path and the KV-R ``all`` path
both captured) runs the same 100-step free-running decode schedule once per arm: ``release/block`` (the reference),
``fast/block``, ``fast/spin`` and ``release/block`` again (run-to-run determinism). Lanes join and leave mid-run (pos 0,
fresh blocks), cross a block boundary (64 tokens), half the lanes sample (seeded top-p) and every tenth step reads the
logits (host path) instead of sampling. Each arm writes its own blocks (attention reads only positions the arm wrote),
so every arm must produce bitwise identical tokens, logprobs and logits on each path. The arms switch the generator's
``host_staging`` / ``waiter`` in place (the persistent-input records are cleared at each switch); nothing is compiled
after the capture.

Run::

    scripts/devrun.sh -t 2400 -n b6a_staging -- python -m pytest models/demos/motif3/tests/test_host_staging_device.py \\
        -s -p no:cacheprovider --timeout=0
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch

from models.demos.motif3.tt.model_config import device_params

N_LAYERS = 4
MAX_MODEL_LEN = 2048
BLOCK = 64
NUM_BLOCKS = 1100
STEPS = 100
WIDTH = MAX_MODEL_LEN // BLOCK
MESH = [pytest.param((4, 8), device_params(), id="4x8")]
ARMS = [("release", "block"), ("fast", "block"), ("fast", "spin"), ("release", "block")]


def _schedule():
    """Per step: {lane: (position, first-token-or-None, blocks)} of the active lanes (the run's own block ids are
    added per arm). Lanes 0, 9, 18, 27 from step 0; 1-7 and 10-12 join at step 5; lane 9 leaves at step 40 and a new
    sequence takes it at step 55; lane 20 joins at step 70."""
    starts = {l: 0 for l in (0, 9, 18, 27)}
    starts.update({l: 5 for l in list(range(1, 8)) + [10, 11, 12]})
    starts[20] = 70
    plan = []
    for s in range(STEPS):
        act = {}
        for lane, s0 in starts.items():
            if lane == 9 and 40 <= s < 55:
                continue
            seq = 1 if (lane == 9 and s >= 55) else 0
            begin = 55 if seq else s0
            if s >= begin:
                act[lane] = (s - begin, seq)
        plan.append(act)
    return plan


def _run(gen, pool, api, path, base_block, plan, vocab):
    """One arm on ``path``: returns per-step records and the host wall time of each step."""
    T = [0.0] * 32
    P = [1.0] * 32
    K = [1] * 32
    S = [None] * 32
    for lane in range(1, 32, 2):  # odd lanes sample (seeded top-p); even lanes greedy
        T[lane], P[lane], K[lane], S[lane] = 1.0, 0.95, 0, 1000 + lane
    sampling = (T, P, K, S)
    last_tok = {}
    recs, walls = [], []
    for step, act in enumerate(plan):
        tok = torch.zeros(32, dtype=torch.int32)
        pos = torch.full((32,), -1, dtype=torch.int32)
        pt = torch.zeros(32, WIDTH, dtype=torch.int32)
        for lane, (p, seq) in act.items():
            key = (lane, seq)
            tok[lane] = last_tok.get(key, 1000 + lane + 7 * seq)
            pos[lane] = p
            first = base_block + 4 * lane + 2 * seq
            pt[lane, : p // BLOCK + 1] = torch.arange(first, first + p // BLOCK + 1, dtype=torch.int32)
        batch = api.DecodeBatch(tokens=tok, positions=pos, page_table=pt)
        lanes = sorted(act)
        t0 = time.perf_counter()
        if step % 10 == 9:
            lg = gen.decode_forward(batch, kv_cache=pool, enable_trace=True, path=path)
            walls.append(time.perf_counter() - t0)
            sel = lg[lanes].clone()
            nxt = sel.float().argmax(-1)
            recs.append(("logits", lanes, sel, nxt))
        else:
            res = gen.decode_forward_sampled(batch, sampling, kv_cache=pool, enable_trace=True, path=path)
            walls.append(time.perf_counter() - t0)
            nxt = res.tokens[lanes].clone()
            recs.append(("sampled", lanes, nxt, res.logprobs[lanes].clone()))
        for i, lane in enumerate(lanes):
            last_tok[(lane, act[lane][1])] = int(nxt[i]) % vocab
    return recs, walls


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@torch.no_grad()
def test_host_staging_and_wait_are_bitwise_neutral(mesh_device, device_params):
    import ttnn
    from models.demos.motif3.tt import generator_api as api
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator, ReplayWaiter
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR
    from models.demos.motif3.tt.weights import HFWeightLoader

    try:
        src = HFWeightLoader()
    except FileNotFoundError as e:
        pytest.skip(f"no local checkpoint: {e}")
    if not all(src.layer_available(l) for l in range(N_LAYERS)):
        pytest.skip("checkpoint layers 0-3 are not local")
    log_fabric(mesh_device, "b6a_staging")
    settings = api.GeneratorSettings(max_batch_size=api.NUM_LANES, max_seq_len=MAX_MODEL_LEN, num_layers=N_LAYERS,
                                     weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=BLOCK)  # fmt: skip
    gen = MotifGenerator.create(hf_config=None, mesh_device=mesh_device, settings=settings)
    failures = []
    try:
        assert gen.host_staging == "fast" and gen.host_wait == "spin" and gen.waiter is not None  # the defaults
        pool = gen.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BLOCK, num_layers=N_LAYERS)
        gen.enable_device_sampling()
        gen._warmed = set(gen.prefill_shapes())  # no prefill in this test: decode-only capture
        gen.extra_decode_paths = [("plain", "all")]
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=WIDTH)
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=WIDTH)
        paths = [("plain", "row"), ("plain", "all")]
        assert set(paths) <= set(gen._paths) and all(gen._paths[k].traced for k in paths)
        plan = _schedule()
        vocab = gen.vocab_size
        out = {}
        run_idx = 0
        for path in paths:
            for arm in ARMS:
                gen.host_staging = arm[0]
                gen.waiter = ReplayWaiter() if arm[1] == "spin" else None
                for p in gen._paths.values():
                    p.host_last.clear()  # the records describe what THIS mode copied
                st0 = dict(gen.stats)
                base = 1 + run_idx * 130  # 4 blocks per lane (2 sequences x 2 blocks): <= 128 per arm
                run_idx += 1
                recs, walls = _run(gen, pool, api, path, base, plan, vocab)
                copies = gen.stats["input_copies"] - st0["input_copies"]
                skipped = gen.stats["input_copies_skipped"] - st0["input_copies_skipped"]
                w = gen.waiter.stats if gen.waiter is not None else {}
                print(f"[b6a] path {path} arm {arm}: {len(recs)} steps, step wall median "
                      f"{statistics.median(walls) * 1e3:.3f} ms, input copies {copies} (+{skipped} skipped), "
                      f"waiter {w}")  # fmt: skip
                out[(path, arm, run_idx)] = recs
        # every arm == the first arm of its path, bitwise
        for path in paths:
            keys = [k for k in out if k[0] == path]
            ref = out[keys[0]]
            for k in keys[1:]:
                got = out[k]
                assert len(got) == len(ref)
                for i, (a, b) in enumerate(zip(ref, got)):
                    if a[0] != b[0] or a[1] != b[1]:
                        failures.append(f"{k} step {i}: kind/lanes differ")
                        break
                    if not all(torch.equal(x, y) for x, y in zip(a[2:], b[2:])):
                        failures.append(f"{k} step {i} ({a[0]}): outputs differ from {keys[0]}")
                        break
        n_sampled = sum(1 for r in out[next(iter(out))] if r[0] == "sampled")
        print(f"[b6a] compared {len(out)} arms x {STEPS} steps ({n_sampled} sampled + {STEPS - n_sampled} logits "
              f"steps each); failures: {failures or 'none'}")  # fmt: skip
        ttnn.synchronize_device(mesh_device)
    finally:
        gen.close()
    assert not failures, "\n".join(failures)
