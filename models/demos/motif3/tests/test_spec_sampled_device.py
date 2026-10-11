# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase G (C10a, ``MOTIF3_SPEC_SAMPLED=on``) on device: the MTP launch's generator (53 layers, ``spec_verify="auto"``,
prefix caching / KV-R, the device sampler), its three traces (T32-spec, the argmax-only T64, the sampled T64), 8 real
prefilled contexts, and a host driver that runs the plugin's protocol (verify ``[anchor, draft]``, the id walk on the
returned ``sampled_ids``, the next draft = the MTP's after the committed token when the MTP consumed it).

* ``test_g_seeded_spec_equals_plain``: seeded requests commit, token for token, what plain sampled decoding (ordinary
  steps only, no drafts) commits: 8 lanes decode plainly; the same 8 contexts / seeds then decode speculatively on 8
  lanes (every draft fits an idle lane: T32 packed verify) and on 24 lanes (3 copies per context, every lane drafting:
  the sampled T64 trace). Configurations: T 1.0 / top-p 0.95 (inverse CDF), T 1.0 / top-p 1 (Gumbel), T 0.6, T 0.8 /
  top-k 20 / top-p 0.8, and a mixed greedy / sampled batch. Also: greedy device draws equal the argmax
  (``greedy_s_ne_a == 0``), and the drafts were exercised (some accepted, some rejected).
* ``test_g_greedy_spec_equals_plain``: the same with greedy lanes (temperature 0): the speculative walk commits the
  plain greedy chain and every T64 step runs on the argmax-only trace.

    KNOB-independent: the module builds its generator with ``spec_sampled=True``. Device job:
    scripts/devrun.sh -t 1800 -n optG_spec_sampled_device -- env OMP_WAIT_POLICY=PASSIVE \\
        python -m pytest -s -p no:cacheprovider --timeout=0 models/demos/motif3/tests/test_spec_sampled_device.py
"""

import math
import time
from typing import Dict, List

import pytest
import torch

from models.demos.motif3.tt import generator_api as api

CTX = 4096
STEPS = 40  # committed tokens per lane (positions CTX .. CTX + STEPS stay in the lane's private block)
N_CTX = 8


def log(m):
    print(f"[g-device {time.strftime('%H:%M:%S')}] {m}", flush=True)


class Session:
    def __init__(self):
        import ttnn

        from models.demos.motif3.reference import golden_stream as gs
        from models.demos.motif3.tt import prefill_plan as PP
        from models.demos.motif3.tt.ccl import log_fabric
        from models.demos.motif3.tt.generator import MotifGenerator
        from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR, open_motif_mesh

        self.ttnn = ttnn
        self.mesh = open_motif_mesh()
        fab = log_fabric(self.mesh, "spec_sampled_device")
        assert fab.get("committed") == "TORUS_XY" and not fab.get("degraded"), fab
        budget = PP.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, api.DEFAULT_PREFILL_ALIGNMENT)
        settings = api.GeneratorSettings(
            max_batch_size=api.NUM_LANES, max_seq_len=32768, num_layers=api.NUM_HIDDEN_LAYERS, kv_cache_dtype="bfp8",
            weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=api.DEFAULT_BLOCK_SIZE,
            weights_source="spec_sampled device test", chunked_prefill=True, prefix_caching=True,
            max_num_batched_tokens=budget, long_prefill_token_threshold=budget, spec_tokens=1, spec_verify="auto",
            spec_sampled=True,
        )  # fmt: skip
        self.gen = gen = MotifGenerator.create(hf_config=None, mesh_device=self.mesh, settings=settings)
        assert gen.cfg.spec_sampled
        bs = self.bs = api.DEFAULT_BLOCK_SIZE
        self.W = W = 32768 // bs
        self.per_lane = math.ceil((CTX + 2) / bs)
        self.uniq0 = 1 + N_CTX * self.per_lane
        num_blocks = max(self.uniq0 + 32 + 16, W + 16)
        self.pool = gen.allocate_kv_cache(num_blocks=num_blocks, block_size=bs, num_layers=gen.num_layers)
        gen.enable_device_sampling(rng_seed=0)
        assert gen.samples_spec and gen.wide_s_path is not None
        base = [i for p in gs.load_prompt_set() for i in p.ids]
        stream = []
        while len(stream) < 1009 * N_CTX + CTX + 64:
            stream += base
        self.stream = stream
        shp = ("sp0", gen.cfg.prefill_bucket(CTX))
        gen.prefill_shapes = lambda: [shp]
        gen.packed_prefill = False
        gen.warmup_prefill(kv_cache=self.pool, enable_trace=False)
        gen.required_prefill_shapes = lambda: []
        gen.warmup_decode(kv_cache=self.pool, enable_trace=False, page_table_width=W)
        gen.warmup_decode(kv_cache=self.pool, enable_trace=True, page_table_width=W)
        assert [k[0] for k in gen.decode_paths()] == ["spec", "wide", "wide_s"], gen.decode_paths()
        for g in range(N_CTX):
            ids = torch.tensor(stream[1009 * g: 1009 * g + CTX], dtype=torch.int32)
            row = torch.zeros(W, dtype=torch.int32)
            nbp = (CTX - 1) // bs + 1
            row[:nbp] = torch.tensor(self.blocks_of(g)[:nbp], dtype=torch.int32)
            gen.prefill_forward(api.PrefillRequest(lane=g, tokens=ids, page_table=row, start=0), kv_cache=self.pool)
        ttnn.synchronize_device(self.mesh)
        log("session ready")

    def blocks_of(self, g):
        return list(range(1 + g * self.per_lane, 1 + (g + 1) * self.per_lane))

    def first_token(self, g):
        return self.stream[1009 * g + CTX]

    def table(self, lanes):
        """Every lane reads its context's shared blocks; the block from CTX on is private to the lane."""
        t = torch.zeros(32, self.W, dtype=torch.int32)
        nb = CTX // self.bs
        for l in lanes:
            t[l, :nb] = torch.tensor(self.blocks_of(l % N_CTX)[:nb], dtype=torch.int32)
            t[l, nb] = self.uniq0 + l
        return t

    def close(self):
        from models.demos.motif3.tt.model_config import close_motif_mesh

        try:
            self.gen.close()
        finally:
            close_motif_mesh(self.mesh)


@pytest.fixture(scope="module")
def session():
    s = Session()
    yield s
    s.close()


def lane_lists(params_of_ctx, lanes):
    T, P, K, S = [0.0] * 32, [1.0] * 32, [1] * 32, [None] * 32
    for l in lanes:
        t, p, k, seed = params_of_ctx(l % N_CTX)
        T[l], P[l], K[l], S[l] = t, p, k, seed
    return T, P, K, S


def decode(s: Session, lanes: List[int], params_of_ctx, *, speculate: bool) -> Dict[int, List[int]]:
    """Commit STEPS tokens per lane; with ``speculate`` every lane drafts whenever the MTP drafted after its committed
    token (the bridge's rule), and verifies run on T32 (drafts fit idle lanes) or the T64 traces."""
    gen, pool = s.gen, s.pool
    sampling = lane_lists(params_of_ctx, lanes)
    tok = {l: s.first_token(l % N_CTX) for l in lanes}
    pos = {l: CTX for l in lanes}
    draft: Dict[int, int] = {}
    out = {l: [] for l in lanes}
    pt = s.table(lanes)
    kinds = []
    acc = rej = 0
    while min(len(v) for v in out.values()) < STEPS:
        tokens = torch.zeros(32, dtype=torch.int32)
        positions = torch.full((32,), -1, dtype=torch.int32)
        drafts = torch.full((32,), -1, dtype=torch.int32)
        for l in lanes:
            tokens[l], positions[l] = tok[l], pos[l]
            if speculate and l in draft:
                drafts[l] = draft[l]
        batch = api.SpecDecodeBatch(tokens=tokens, positions=positions, draft_tokens=drafts, page_table=pt)
        res = gen.decode_forward_spec(batch, kv_cache=pool, enable_trace=True, want_logits=False, sampling=sampling)
        kinds.append(gen.last_verify_kind)
        new_draft = {}
        for l in lanes:
            if batch.is_verify:
                x = int(res.sampled_ids[l, 0])
            else:
                x = int(res.sample.tokens[l])
            if int(drafts[l]) >= 0 and x == int(drafts[l]):
                y = int(res.sampled_ids[l, 1])
                committed, col = [x, y], 1
                acc += 1
            else:
                committed, col = [x], 0
                rej += int(int(drafts[l]) >= 0)
            out[l] += committed
            tok[l] = committed[-1]
            pos[l] += len(committed)
            if int(res.mtp_input[l, col]) == committed[-1]:
                new_draft[l] = int(res.mtp_argmax[l, col])
        draft = new_draft if speculate else {}
    log(f"lanes {lanes[0]}..{lanes[-1]} speculate={speculate}: steps {len(kinds)}, kinds "
        f"{ {k: kinds.count(k) for k in set(kinds)} }, drafts accepted {acc} rejected {rej}")  # fmt: skip
    s.last = {"kinds": kinds, "acc": acc, "rej": rej}
    return {l: v[:STEPS] for l, v in out.items()}


CONFIGS = {
    "T1.0_p0.95": lambda g: (1.0, 0.95, 0, 1000 + g),
    "T1.0_gumbel": lambda g: (1.0, 1.0, 0, 2000 + g),
    "T0.6": lambda g: (0.6, 1.0, 0, 3000 + g),
    "T0.8_k20_p0.8": lambda g: (0.8, 0.8, 20, 4000 + g),
    "mixed": lambda g: (0.0, 1.0, 1, None) if g % 2 == 0 else (1.0, 0.95, 0, 5000 + g),
}


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_g_seeded_spec_equals_plain(session, cfg):
    s, params = session, CONFIGS[cfg]
    gen = s.gen
    plain = decode(s, list(range(0, 8)), params, speculate=False)
    t32 = decode(s, list(range(8, 16)), params, speculate=True)
    k32 = dict(s.last)
    wide = decode(s, list(range(8, 32)), params, speculate=True)
    kw = dict(s.last)
    for g in range(8):
        ref = plain[g]
        assert t32[8 + g] == ref, f"{cfg} ctx {g}: T32 spec diverged at {_first_diff(t32[8 + g], ref)}"
        for c in (8, 16, 24):
            assert wide[c + g] == ref, f"{cfg} ctx {g} lane {c + g}: T64 spec diverged at {_first_diff(wide[c + g], ref)}"
    assert "spec" in k32["kinds"] and k32["acc"] + k32["rej"] > 0
    assert "wide_s" in kw["kinds"], kw["kinds"]  # every configuration has sampled rows
    assert kw["acc"] > 0, "no draft was accepted: the drafts were not exercised"
    assert gen.stats["greedy_s_ne_a"] == 0
    log(f"{cfg}: identical over {8 * STEPS} plain tokens x 4 speculative copies; sampling stats {gen.sampling_stats()}")


def test_g_greedy_spec_equals_plain(session):
    s = session
    params = lambda g: (0.0, 1.0, 1, None)  # noqa: E731
    plain = decode(s, list(range(0, 8)), params, speculate=False)
    wide = decode(s, list(range(8, 32)), params, speculate=True)
    kw = dict(s.last)
    for g in range(8):
        for c in (8, 16, 24):
            assert wide[c + g] == plain[g], f"greedy ctx {g} lane {c + g}: diverged at {_first_diff(wide[c + g], plain[g])}"
    assert "wide" in kw["kinds"] and "wide_s" not in kw["kinds"], kw["kinds"]
    assert kw["acc"] > 0
    assert s.gen.stats["greedy_s_ne_a"] == 0


def _first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
