# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profiling helper, not a correctness test (SKIPS unless ``SOLAR_OPEN_PERF_PROFILE=1``): the two attention-prefill
items of the phase-3 profile at the real Solar-Open shapes (TP=8: 8 q heads + 1 kv head of head_dim 128 per device,
hidden 4096) -- (a) causal SDPA at 1K / 8K tokens over q/k chunk sizes and core grids (one device), (b) the TP
all_reduce of the [1, 1, S, 4096] attention / MoE partial at 1K / 8K in bf16 and bfloat8_b (1x8 mesh).

Every configuration runs REPS times between "mb_<name>_start" / "mb_<name>_stop" signposts so the tracy ops CSV gives
its device kernel duration; ``SOLAR_OPEN_PERF_OUT`` receives the host walls and the PCC of every SDPA variant against
the production chunk config.

    SOLAR_OPEN_PERF_PROFILE=1 SOLAR_OPEN_PERF_OUT=/path/mb.json python -m tracy -r -p -v -m pytest \
        models/demos/solar_open/tests/perf/test_attention_prefill_microbench.py -x -p no:cacheprovider
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.solar_open.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.solar_open.tt.attention_configs import SolarOpenAttentionProgramConfig

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(header, message=None):
        logger.info(f"SIGNPOST {header}")


PERF_OUT = os.getenv("SOLAR_OPEN_PERF_OUT", "")
PROFILE = os.getenv("SOLAR_OPEN_PERF_PROFILE", "") == "1"
REPS = int(os.getenv("SOLAR_OPEN_PERF_REPS", "6"))
Q_HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 8, 1, 128, 4096  # per device at TP=8


class Bench:
    def __init__(self, device, key):
        self.device = device
        self.key = key
        self.results = {}
        self.failures = {}

    def run(self, name, fn, reps=REPS, keep=False):
        """fn() -> output tensor. Compile run, then `reps` signposted runs; returns the last output when keep."""
        try:
            out = fn()
            ttnn.synchronize_device(self.device)
            if out is not None:
                out.deallocate(True)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
            logger.warning(f"[mb {name}] FAILED: {msg}")
            self.failures[name] = msg
            return None
        walls = []
        last = None
        signpost(f"mb_{name}_start")
        for i in range(reps):
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(self.device)
            walls.append((time.perf_counter() - t0) * 1e3)
            if out is not None and (not keep or i < reps - 1):
                out.deallocate(True)
            else:
                last = out
        signpost(f"mb_{name}_stop")
        self.results[name] = {"wall_ms_min": min(walls), "wall_ms_mean": sum(walls) / len(walls), "reps": reps}
        logger.info(f"[mb {name}] wall min {min(walls):.3f} ms mean {sum(walls) / len(walls):.3f} ms")
        return last

    def note(self, name, key, value):
        self.results.setdefault(name, {})[key] = value
        logger.info(f"[mb {name}] {key} = {value}")

    def dump(self):
        if not PERF_OUT:
            return
        data = {}
        if os.path.isfile(PERF_OUT):
            data = json.loads(open(PERF_OUT).read())
        data[self.key] = {"results": self.results, "failures": self.failures}
        with open(PERF_OUT, "w") as f:
            json.dump(data, f, indent=2)


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(1, 1)])
def test_sdpa_prefill_microbench(mesh_device, device_params, reset_seeds):
    """Causal SDPA [1, 8, S, 128] x [1, 1, S, 128] (GQA, bf16) at S = 1024 / 8192: q/k chunk sizes x core grids."""
    if not PROFILE:
        pytest.skip(
            "profiling helper: set SOLAR_OPEN_PERF_PROFILE=1 (and run under `python -m tracy -r -p -v -m pytest ...`)"
        )
    device = mesh_device
    bench = Bench(device, "sdpa_prefill")
    pc = SolarOpenAttentionProgramConfig()
    compute = pc.get_compute_kernel_config()
    full_grid = device.compute_with_storage_grid_size()
    bench.note("meta", "compute_grid", [full_grid.x, full_grid.y])
    bench.note(
        "meta",
        "production_chunks",
        {
            "small": [pc.prefill_q_chunk_size_small, pc.prefill_k_chunk_size_small],
            "large": [pc.prefill_q_chunk_size_large, pc.prefill_k_chunk_size_large],
            "threshold": pc.prefill_threshold,
        },
    )
    g = torch.Generator().manual_seed(11)
    for S in (1024, 8192):
        q = ttnn.from_torch(
            torch.randn(1, Q_HEADS, S, HEAD_DIM, generator=g),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        k = ttnn.from_torch(
            torch.randn(1, KV_HEADS, S, HEAD_DIM, generator=g),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        v = ttnn.from_torch(
            torch.randn(1, KV_HEADS, S, HEAD_DIM, generator=g),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        prod = pc.get_prefill_sdpa_config(device, S)
        variants = [
            (
                "prod",
                (prod.compute_with_storage_grid_size.x, prod.compute_with_storage_grid_size.y),
                prod.q_chunk_size,
                prod.k_chunk_size,
            )
        ]
        for qc, kc in ((64, 64), (128, 128), (256, 256), (512, 512), (128, 256), (256, 512), (512, 256), (256, 128)):
            variants.append((f"g8x8_q{qc}_k{kc}", (8, 8), qc, kc))
        for qc, kc in ((128, 128), (256, 256), (512, 512), (256, 512)):
            variants.append((f"gfull_q{qc}_k{kc}", (full_grid.x, full_grid.y), qc, kc))
        ref = None
        for tag, grid, qc, kc in variants:
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                exp_approx_mode=False,
                q_chunk_size=qc,
                k_chunk_size=kc,
            )
            name = f"sdpa_S{S}_{tag}"
            out = bench.run(
                name,
                lambda: ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=True, program_config=cfg, compute_kernel_config=compute
                ),
                keep=True,
            )
            if out is None:
                continue
            out_t = ttnn.to_torch(out)
            out.deallocate(True)
            if ref is None:
                ref = out_t
            else:
                ok, pcc = comp_pcc(ref, out_t, 0.999)
                bench.note(name, "pcc_vs_prod", float(str(pcc).split()[-1]) if not isinstance(pcc, float) else pcc)
        q.deallocate(True)
        k.deallocate(True)
        v.deallocate(True)
    bench.dump()
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(1, 8)])
def test_allreduce_prefill_microbench(mesh_device, device_params, reset_seeds):
    """ttnn.all_reduce (ring, cluster axis 1) of the [1, 1, S, 4096] partial at S = 1024 / 8192 in bf16 / bfloat8_b,
    the collective every layer runs twice per prefill (attention o_proj partial, MoE partial)."""
    if not PROFILE:
        pytest.skip(
            "profiling helper: set SOLAR_OPEN_PERF_PROFILE=1 (and run under `python -m tracy -r -p -v -m pytest ...`)"
        )
    if tuple(mesh_device.shape) != (1, 8):
        pytest.skip("sized for the 1x8 mesh")
    bench = Bench(mesh_device, "allreduce_prefill")
    g = torch.Generator().manual_seed(12)
    for S in (1024, 8192):
        for dtype, dname in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bfp8")):
            x_t = torch.randn(1, 1, S, HIDDEN, generator=g)
            x = ttnn.from_torch(
                x_t,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for links in (1, 2):
                name = f"allreduce_S{S}_{dname}_links{links}"
                bench.run(
                    name, lambda: ttnn.all_reduce(x, num_links=links, topology=ttnn.Topology.Ring, cluster_axis=1)
                )
            # the pair the composite runs: reduce_scatter (dim 3) + all_gather (dim 3), timed separately
            name = f"reduce_scatter_S{S}_{dname}"
            rs = bench.run(
                name,
                lambda: ttnn.reduce_scatter(x, dim=3, num_links=1, topology=ttnn.Topology.Ring, cluster_axis=1),
                keep=True,
            )
            if rs is not None:
                name = f"all_gather_S{S}_{dname}"
                bench.run(
                    name, lambda: ttnn.all_gather(rs, dim=3, num_links=1, topology=ttnn.Topology.Ring, cluster_axis=1)
                )
                rs.deallocate(True)
            x.deallocate(True)
    bench.dump()
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
    except Exception:
        pass
