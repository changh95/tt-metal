# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of the exact Motif mHC coefficient kernel (``tt/kernels/sinkhorn_motif.py``, WAVE_A_REVIEW MHC-6).

Run ONLY through the device lock (4x8 mesh, FABRIC_2D_TORUS_XY; ~35 s for the whole file):

    scripts/devrun.sh -t 1500 -n sinkhorn_motif -- \
        python -m pytest models/demos/motif3/tests/unit/test_sinkhorn_motif.py -s -p no:cacheprovider

Real mHC mixes (layers 0-5 and 28-35, both sites, one 4096-token real context) come from a host-only capture with
the CPU reference (WAVE_A_REVIEW Appendix A recipe; ~4 min; 22 MB under tt_cache/test/sinkhorn_kernel/):

    scripts/hostrun.sh -n sinkhorn_capture -- python -m models.demos.motif3.tests.unit.test_sinkhorn_motif --capture

``test_real_mixes`` skips (never captures) if that file is missing; ``test_row_varying_decode`` and the multi-site part
of ``test_trace_replay`` then use synthetic lanes. Host-only tests (pure torch / ttnn config objects, no tensors):

    scripts/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib -q \
        models/demos/motif3/tests/unit/test_sinkhorn_motif.py -k "host_consts or import_is"

The out-of-tree kernels get no tt-metal CI; after an uplift, JIT-compile every variant host-only on a mock device
(~10 s, no lock):

    TT_METAL_MOCK_CLUSTER_DESC_PATH=$PWD/tt_metal/third_party/tt-cluster-descriptors/blackhole/p100_cluster_desc/\
p100_cluster_desc.yaml scripts/hostrun.sh -- python -m models.demos.motif3.tests.unit.test_sinkhorn_motif --mock-compile

Accuracy bounds: ``H_TOL`` (1e-5 on H) and ``HPP_TOL`` (1e-6 on h_pre / h_post) are **self-imposed** fp32-exactness
targets for this kernel (measured <= 4.5e-7 / 1.8e-7). WAVE_A_REVIEW MHC-6 sets no number; the acceptance of the mHC
module (MHC-5, README §12) is max|dH| <= 5e-3 and h_pre / h_post <= 2e-3.

Latency: traced figures are quoted as the raw per-call time ``t(n) / n`` of a trace of ``n`` back-to-back calls (an
upper bound: it includes ``sync / n``, ~140 us / n); the gates' slope ``(t(n) - t(n/2)) / (n/2)`` is printed too, but
for calls of a few us it is at the noise level.

Tests (every device test logs the committed fabric first; accuracy cases also check replica bit identity):
* ``test_host_consts_and_golden`` (host): consts round trip; the golden is bitwise the reference's MHCLayer / sinkhorn
  math; the compute config equals the ``mhc`` role; ``MotifSinkhorn(cfg=)`` refuses an h_post coefficient != 1;
  output-geometry and device-identity helpers.
* ``test_import_is_self_contained`` (host): ``tt.kernels`` and ``tt.kernels.sinkhorn_motif`` import no other
  ``models.*`` or heavy package and open no device (README §13; same probe as ``test_infra_import``).
* ``test_layout_passthrough``: mode "passthrough" routes the raw mixes through the transposes / DST layout / pack
  path -> outputs equal the input columns bit for bit (T = 8, 32, 100, 4096; incl. +-1e-30, +-3.4e38).
* ``test_logits_exact``: mode "logits": clamp(alpha * p + b) bitwise equal to torch fp32 (zero differing elements;
  a re-fused multiply-add would differ), synthetic and real mixes.
* ``test_synthetic_regimes``: full mode vs the fp32 / fp64 Motif golden on the G3 logit regimes (realistic, moderate,
  wide, peaked, degenerate rows where the 1e-8 floor binds, +90 overflow) at T = 8 / 32 / 4096.
* ``test_real_mixes``: real mixes of 28 sites at T = 8 / 32 (the most extreme tokens of each site) and T = 4096 (all)
  vs the reference's exact maps; the golden also equals the reference module's own h_pre / h_post / h_res taps.
* ``test_row_varying_decode``: the README §12 decode setup: each DP row gets different (real) lanes
  (``rope.lanes_to_rows`` + ``rope.shard_lanes``, replicated over TP); every chip vs the golden of its row's lanes
  ``8 dp .. 8 dp + 7``, ``ccl.replicas_identical(o, mesh, "tp")``, rows really differ; stock and wrnc layouts, eager,
  mhc.py's ``[1, 8, 32]`` view of a garbage-padded tile, and one trace replayed with new lanes per row.
* ``test_latency``: eager / traced / host-enqueue cost per call at T = 8 / 32 / 4096 vs the stock
  ``mhc_split_sinkhorn`` (G3); asserts faster than stock traced (raw) at T <= 32.
* ``test_prefill_latency``: stock vs wrnc layout at the prefill buckets T = 4096 / 8192 / 16384 / 32768 (preallocated
  outputs; eager warmed and traced), the number tt/mhc.py needs before using wrnc in prefill.
* ``test_token_counts_and_source``: T in 1 .. 32768 (partial tiles, both num_halves paths, uneven core split, the
  8192 / 16384 / 32768 prefill buckets; wrnc at 8192 and 32768), ``MotifSinkhorn.from_source`` with and without
  ``cfg``, 4D / width-24 / L1 inputs and outputs, ``reuse_outputs``.
* ``test_input_views_and_validation``: NaN / Inf in the padding rows and columns 24..31 (mhc.py's decode view), a view
  whose padded height exceeds ceil(T/32) tiles (read and written as ceil(T/32) tiles), every invalid input / output
  raises (also for a tensor at a recycled buffer address, i.e. on a descriptor-cache hit), interleaved layouts /
  modes / options on the same tensors, iters 0 / 1 / 5 / 40, ``max_cores``, L1 consts.
* ``test_trace_replay``: decode usage: prepare + one eager warm-up, capture once, replay with new inputs copied into the
  persistent mixes tensor (T = 8, 32); a 3-site trace mixing the stock and wrnc layouts, replayed with new inputs.
* ``test_host_overhead_breakdown``: host cost of an eager call by component (allocation, key, validation, dispatch).
* ``test_wrnc_layout``: layout="wrnc" equals the stock outputs re-laid out (bitwise), feeds
  ``attn_res_weighted_reduce_nc`` (x_red and the fused post) correctly, and its traced cost vs stock + glue.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn

ROOT = Path("/home/ttuser/hchang/experiments/motif-3")
CAPTURE_DIR = ROOT / "tt_cache" / "test" / "sinkhorn_kernel"
CAPTURE_FILE = CAPTURE_DIR / "real_mixes_L0-5_28-35_T4096.pt"
REAL_LAYERS = tuple(range(0, 6)) + tuple(range(28, 36))
SITES = ("mhc_attn", "mhc_ffn")
N_TOKENS = 4096
TRACE = 64 * 1024 * 1024
# Self-imposed fp32-exactness targets of this kernel (measured <= 4.5e-7 / 1.8e-7). MHC-6 sets no number; the mHC
# module acceptance (MHC-5, README §12) is max|dH| <= 5e-3, h_pre / h_post <= 2e-3.
H_TOL = 1e-5
HPP_TOL = 1e-6
PREFILL_BUCKETS = (4096, 8192, 16384, 32768)
TILE_ROWS = 32  # rows of one tile (mhc.py's decode view: 8 logical lanes of a 32-row tile)


# =====================================================================================================================
# host-only capture of real mHC mixes (CPU reference; run through scripts/hostrun.sh)
# =====================================================================================================================
def capture_real_mixes(out_path: Path = CAPTURE_FILE, layers=REAL_LAYERS, n_tokens: int = N_TOKENS) -> Path:
    """WAVE_A_REVIEW Appendix A recipe on one real 4096-token context: the 6 chat prompts of
    ``reference/prompts/messages.json`` rendered with the Motif template and concatenated (repeated to 4096).
    Saves, per site ``"{layer}.{site}"``: the raw projections ``p [T, 24]`` fp32 (tap ``mixes``), the site's fp32
    alphas / biases, and the reference module's own ``h_pre / h_post / h_res`` taps."""
    from models.demos.motif3.reference.golden import capture_model_goldens
    from models.demos.motif3.reference.tokenizer import encode_chat, load_tokenizer
    from models.demos.motif3.reference.weights import load_reference_model

    t0 = time.time()
    prompts_file = Path(__file__).resolve().parents[2] / "reference" / "prompts" / "messages.json"
    prompts = json.loads(prompts_file.read_text())["prompts"]
    tok = load_tokenizer()
    ids, spans = [], []
    for p in prompts:
        seg = encode_chat(p["messages"], tok, add_generation_prompt=bool(p.get("add_generation_prompt", True)))
        spans.append((p["name"], len(ids), len(ids) + len(seg)))
        ids += seg
    n_unique = len(ids)
    while len(ids) < n_tokens:
        ids += ids[: n_tokens - len(ids)]
    ids = ids[:n_tokens]
    print(f"[capture] {n_unique} unique prompt tokens -> {len(ids)} tokens; prompts {spans}", flush=True)

    model = load_reference_model(layer_ids=range(max(layers) + 1), dtype=torch.bfloat16, lazy_experts=True)
    want = {f"layers.{l}.{s}.{k}" for l in layers for s in SITES for k in ("mixes", "h_pre", "h_post", "h_res")}
    print(f"[capture] model loaded in {time.time() - t0:.0f}s; running {max(layers) + 1} layers", flush=True)
    with torch.no_grad():
        g = capture_model_goldens(model, torch.tensor([ids]), None, include=lambda n: n in want)
    pre = g["prefill"]
    sites = {}
    for l in layers:
        for s in SITES:
            m = getattr(model.model.layers[str(l)], s)
            f32 = lambda t: t.detach().to(torch.float32).reshape(-1).clone()  # noqa: E731
            sites[f"{l}.{s}"] = dict(
                p=pre[f"layers.{l}.{s}.mixes"].to(torch.float32).reshape(-1, 24).clone(),
                alpha_pre=f32(m.alpha_pre),
                alpha_post=f32(m.alpha_post),
                alpha_res=f32(m.alpha_res),
                bias_pre=f32(m.bias_pre),
                bias_post=f32(m.bias_post),
                bias_res=f32(m.bias_res),  # row-major 4i + j
                ref_h_pre=pre[f"layers.{l}.{s}.h_pre"].to(torch.float32).reshape(-1, 4).clone(),
                ref_h_post=pre[f"layers.{l}.{s}.h_post"].to(torch.float32).reshape(-1, 4).clone(),
                ref_h_res=pre[f"layers.{l}.{s}.h_res"].to(torch.float32).reshape(-1, 16).clone(),
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(
        recipe="WAVE_A_REVIEW Appendix A (load_reference_model bf16, lazy experts, capture_model_goldens prefill)",
        tokens=len(ids),
        unique_tokens=n_unique,
        prompts=spans,
        layers=list(layers),
        seconds=round(time.time() - t0, 1),
    )
    torch.save(dict(meta=meta, sites=sites), out_path)
    print(f"[capture] saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s", flush=True)
    return out_path


def _load_capture():
    if not CAPTURE_FILE.is_file():
        pytest.skip(f"real mHC mixes not captured ({CAPTURE_FILE}); run the --capture entry point via hostrun.sh")
    return torch.load(CAPTURE_FILE, weights_only=True)


def _capture_or_none():
    return torch.load(CAPTURE_FILE, weights_only=True) if CAPTURE_FILE.is_file() else None


def mock_compile() -> int:
    """JIT-compile every kernel variant on a MOCK Blackhole device (no chips, no device lock): the out-of-tree kernels
    get no tt-metal CI, so run this after a tt-metal uplift. Needs TT_METAL_MOCK_CLUSTER_DESC_PATH (see __main__)."""
    from models.demos.motif3.tt.kernels import sinkhorn_motif as SM

    if not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        raise SystemExit("set TT_METAL_MOCK_CLUSTER_DESC_PATH to a BH cluster descriptor (see the module docstring)")
    dev = ttnn.open_device(device_id=0)
    failures = 0
    try:
        consts = SM.consts_to_device(SM.build_consts(0.1, 0.2, 0.3, [0.0] * 4, [0.0] * 4, [0.0] * 16), dev)
        for T, layout, mode, refine in [
            (8, "stock", "full", False),
            (64, "stock", "full", False),
            (64, "stock", "full", True),
            (32, "stock", "passthrough", False),
            (32, "stock", "logits", False),
            (8, "wrnc", "full", False),
            (64, "wrnc", "full", False),
        ]:
            mix = ttnn.from_torch(torch.zeros(T, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            outs = SM.allocate_outputs(mix, layout=layout)
            t0 = time.time()
            try:
                desc = SM.program_descriptor(mix, consts, *outs, mode=mode, div_refine=refine, layout=layout)
                ttnn.experimental.prepare_generic_op([mix, consts, *outs], desc)
                print(f"[mock-compile] T={T} layout={layout} mode={mode} refine={refine}: OK ({time.time() - t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"[mock-compile] T={T} layout={layout} mode={mode} refine={refine}: FAILED {e}")
    finally:
        ttnn.close_device(dev)
    return failures


if __name__ == "__main__":
    # host-only entry points (run through scripts/hostrun.sh, devices hidden):
    #   --capture       real mHC mixes for test_real_mixes (CPU reference, ~4 min)
    #   --mock-compile  JIT build of every kernel variant on a mock device, e.g.
    #     TT_METAL_MOCK_CLUSTER_DESC_PATH=$PWD/tt_metal/third_party/tt-cluster-descriptors/blackhole/p100_cluster_desc/\
    #     p100_cluster_desc.yaml scripts/hostrun.sh -- python -m models.demos.motif3.tests.unit.test_sinkhorn_motif --mock-compile
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true", help="capture real mHC mixes on CPU (host-only)")
    ap.add_argument("--mock-compile", action="store_true", help="JIT-compile all kernel variants on a mock device")
    args = ap.parse_args()
    if args.capture:
        torch.set_num_threads(int(os.environ.get("MOTIF3_CAPTURE_THREADS", "32")))
        capture_real_mixes()
    if args.mock_compile:
        raise SystemExit(1 if mock_compile() else 0)


# =====================================================================================================================
# device helpers
# =====================================================================================================================
def _mesh_params():
    from models.demos.motif3.tt.model_config import device_params

    return [pytest.param((4, 8), device_params("FABRIC_2D_TORUS_XY", TRACE), id="4x8-torus2d")]


MESH = _mesh_params()


def _sm():
    from models.demos.motif3.tt.kernels import sinkhorn_motif as SM

    return SM


def _log_fabric(mesh_device, tag):
    from models.demos.motif3.tt.ccl import log_fabric

    return log_fabric(mesh_device, tag)


def _to_mesh(t: torch.Tensor, mesh_device, memory_config=None, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.to(torch.float32).contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _read(t, idx: int = 0) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[idx]).to(torch.float32)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32)


def _replicas_identical(t) -> bool:
    shards = ttnn.get_device_tensors(t)
    ref = _bits(ttnn.to_torch(shards[0]).to(torch.float32))
    return all(torch.equal(_bits(ttnn.to_torch(s).to(torch.float32)), ref) for s in shards[1:])


def _free(o):
    if isinstance(o, (list, tuple)):
        for x in o:
            _free(x)
    elif isinstance(o, ttnn.Tensor):
        try:
            ttnn.deallocate(o)
        except Exception:
            pass


def _maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.double() - b.double()).abs()
    return float(d.max()) if bool(torch.isfinite(d).all()) else float("inf")


class _Capture:
    """Exception-safe trace capture (a dangling capture once hung close_mesh_device; GATES_RESULTS.md §11.6)."""

    def __init__(self, mesh_device):
        self.mesh, self.tid = mesh_device, None

    def __enter__(self):
        self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            ttnn.end_trace_capture(self.mesh, self.tid, cq_id=0)
        except Exception:
            if exc_type is None:
                raise
        if exc_type is not None:
            try:
                ttnn.release_trace(self.mesh, self.tid)
            except Exception:
                pass
        return False


def _trace_min_us(mesh_device, fn, n, reps):
    with _Capture(mesh_device) as cap:
        for _ in range(n):
            _free(fn())
    try:
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            times.append((time.perf_counter() - t0) * 1e6)
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    return min(times)


def _traced_us(mesh_device, fn, n=32, reps=9):
    """``(slope, raw)`` us per call of a trace of ``n`` back-to-back calls: slope = (min t(n) - min t(n/2)) / (n/2)
    (the gates' method), raw = min t(n) / n (an upper bound: includes sync / n; synchronize alone costs ~140 us on this
    mesh). Quote raw for calls of a few us (the slope is then at the noise level)."""
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    t1 = _trace_min_us(mesh_device, fn, n // 2, reps)
    t2 = _trace_min_us(mesh_device, fn, n, reps)
    return max((t2 - t1) / (n - n // 2), 0.0), t2 / n


def _eager_us(mesh_device, fn, iters=20, warmup=2):
    for _ in range(warmup):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


# synthetic constants / regimes (as gate G3: |alpha| <= 0.33, biases within +-0.55, B_res within +-0.48)
ALPHA = (-0.22, 0.33, 0.18)


def _make_consts(seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    b_pre = (torch.rand(4, generator=g) - 0.5) * 1.0
    b_post = (torch.rand(4, generator=g) - 0.5) * 1.0
    b_res = (0.17 * torch.randn(16, generator=g)).clamp(-0.48, 0.48)
    return _sm().build_consts(ALPHA[0], ALPHA[1], ALPHA[2], b_pre, b_post, b_res)


def _make_p(T: int, regime: str, seed: int) -> torch.Tensor:
    """Raw projections [T, 24] whose logits alpha p + b have the G3 regime's spread."""
    g = torch.Generator().manual_seed(seed)
    std = {"realistic": 0.5, "moderate": 3.0, "wide": 10.0, "peaked": 30.0, "degenerate": 0.5, "overflow": 0.5}[regime]
    alpha = torch.tensor([ALPHA[0]] * 4 + [ALPHA[1]] * 4 + [ALPHA[2]] * 16)
    p = torch.randn(T, 24, generator=g) * std / alpha.abs()
    if regime == "degenerate":  # every 4th token: row 1 of M has four logits ~ -25 (row sum < 1e-8 -> floor binds)
        idx = torch.arange(0, T, 4)
        p[idx[:, None], 12 + torch.arange(4)[None, :]] = -25.0 / ALPHA[2]
    if regime == "overflow":  # every 4th token: one res logit at +90
        idx = torch.arange(0, T, 4)
        p[idx, 13] = 90.0 / ALPHA[2]
    return p


def _pad32(p: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(p.shape[0], 32, dtype=torch.float32)
    out[:, : p.shape[1]] = p
    return out


def _compare(p: torch.Tensor, consts: torch.Tensor, dev) -> dict:
    SM = _sm()
    g32 = SM.motif_mhc_maps_torch(p, consts)
    g64 = SM.motif_mhc_maps_torch(p, consts, dtype=torch.float64)
    return dict(
        H=_maxabs(g32[2], dev[2]),
        h_pre=_maxabs(g32[0], dev[0]),
        h_post=_maxabs(g32[1], dev[1]),
        H_vs_fp64=_maxabs(g64[2], dev[2]),
        golden32_vs_fp64=_maxabs(g64[2], g32[2]),
        colsum_dev=float((dev[2].reshape(-1, 4, 4).sum(-2) - 1).abs().max()),
    )


def _compare_wrnc(p: torch.Tensor, consts: torch.Tensor, w_pre: torch.Tensor, w_post: torch.Tensor) -> dict:
    SM = _sm()
    e_pre, e_post = SM.wrnc_weights_torch(*SM.motif_mhc_maps_torch(p, consts))
    return dict(w_pre=_maxabs(e_pre, w_pre.reshape(e_pre.shape)), w_post=_maxabs(e_post, w_post.reshape(e_post.shape)))


def _run(mesh_device, p: torch.Tensor, consts_tt, **kw):
    mix = _to_mesh(_pad32(p), mesh_device)
    outs = _sm().motif_sinkhorn(mix, consts_tt, **kw)
    host = tuple(_read(o) for o in outs)
    return mix, outs, host


def _site_consts(site: dict) -> torch.Tensor:
    return _sm().build_consts(
        site["alpha_pre"], site["alpha_post"], site["alpha_res"], site["bias_pre"], site["bias_post"], site["bias_res"]
    )


def _sites(names, n_synthetic: int = 1024):
    """``[(name, p [N, 24], consts [64, 32])]``: real capture sites, or synthetic stand-ins if not captured."""
    cap = _capture_or_none()
    if cap is not None:
        return [(n, cap["sites"][n]["p"], _site_consts(cap["sites"][n])) for n in names]
    regimes = ("wide", "peaked", "degenerate", "overflow", "moderate")
    return [
        (f"synthetic{k}", _make_p(n_synthetic, regimes[k % len(regimes)], 4000 + k), _make_consts(60 + k))
        for k in range(len(names))
    ]


# =====================================================================================================================
# host-only tests
# =====================================================================================================================
def test_host_consts_and_golden():
    """Pure torch / ttnn config objects (no device): build_consts round trip, golden == reference.modules (sinkhorn
    and MHCLayer maps), compute config == the mhc role, MotifSinkhorn(cfg=) coefficient guard, geometry helpers."""
    SM = _sm()
    c = _make_consts(3)
    s = SM.scalars_from_consts(c)
    assert torch.equal(SM.consts_from_scalars(s), c)
    p = _make_p(256, "wide", 5)
    h_pre, h_post, H = SM.motif_mhc_maps_torch(p, c)
    from models.demos.motif3.reference import modules as R

    l_pre, l_post, l_res = SM.motif_logits_torch(p, c)
    assert torch.equal(H.reshape(-1, 4, 4), R.sinkhorn(l_res, 20))
    mhc = R.MHCLayer(4, 8)
    with torch.no_grad():
        for k in ("alpha_pre", "alpha_post", "alpha_res"):
            getattr(mhc, k).copy_(s[k].reshape(1))
        mhc.bias_pre.copy_(s["bias_pre"])
        mhc.bias_post.copy_(s["bias_post"])
        mhc.bias_res.copy_(s["bias_res"].reshape(4, 4))
        # MHCLayer.forward computes p itself; replicate its coefficient math on our p
        rp = torch.sigmoid((mhc.alpha_pre * p[:, :4] + mhc.bias_pre).clamp(-10.0, 10.0))
        rq = mhc.h_post_coeff * torch.sigmoid((mhc.alpha_post * p[:, 4:8] + mhc.bias_post).clamp(-10.0, 10.0))
        rr = R.sinkhorn(mhc.alpha_res * p[:, 8:].reshape(-1, 4, 4) + mhc.bias_res, mhc.sinkhorn_iters)
    assert torch.equal(rp, h_pre) and torch.equal(rq, h_post) and torch.equal(rr.reshape(-1, 16), H)
    assert float(mhc.h_post_coeff) == SM.H_POST_COEFF and int(mhc.sinkhorn_iters) == SM.DEFAULT_ITERS

    # README §4: the kernel's ComputeConfigDescriptor (needed for the per-CB UnpackToDestFp32 of generic_op) carries
    # exactly the values of the mhc role; a retuned role must be reviewed against the kernel's requirements.
    from models.demos.motif3.tt.model_config import COMPUTE_ROLES

    role = COMPUTE_ROLES["mhc"]
    assert (role.fidelity, role.fp32_acc, role.approx) == (SM.COMPUTE_FIDELITY, SM.COMPUTE_FP32_ACC, SM.COMPUTE_APPROX)
    cc = SM._compute_config()
    assert cc.math_fidelity == getattr(ttnn.MathFidelity, role.fidelity)
    assert bool(cc.fp32_dest_acc_en) == role.fp32_acc and bool(cc.math_approx_mode) == role.approx

    # MotifSinkhorn(cfg=...) refuses an h_post coefficient the kernel does not implement (before any device work)
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="h_post coefficient"):
        SM.MotifSinkhorn(None, c, cfg=SimpleNamespace(mhc_h_post_coeff=2.0, sinkhorn_iters=20))

    # output geometry (what the writers address) and device identity
    assert SM.output_shapes(40, "stock") == ([40, 4], [40, 4], [40, 16])
    assert SM.output_shapes(40, "wrnc") == ([1, 4, 40, 1], [4, 5, 40, 1])
    assert SM._tile_padded([4, 5, 40, 1]) == [4, 5, 64, 32] and SM._tile_padded([8, 16]) == [32, 32]
    assert SM._pages_written(2, "wrnc") == (8, 40) and SM._pages_written(3, "stock") == (3, 3, 3)

    class _Mesh:  # MeshDevice.id() is a process-wide counter: a reopened mesh gets a new id
        def id(self):
            return 7

    class _Legacy:  # no id(): Python id() (reusable after a close) plus the compute grid
        def compute_with_storage_grid_size(self):
            return SimpleNamespace(x=12, y=10)

    legacy = _Legacy()
    assert SM._device_uid(_Mesh()) == 7 and SM._device_uid(legacy) == ("py", id(legacy), 12, 10)


_IMPORT_PROBE = r"""
import importlib, json, sys
mods = json.loads(sys.argv[1])
for m in mods:
    importlib.import_module(m)
demos = sorted(m for m in sys.modules if m.startswith("models.demos.") and not m.startswith("models.demos.motif3"))
other_models = sorted(m for m in sys.modules if m.startswith("models.") and not m.startswith("models.demos"))
heavy = sorted(m for m in ("vllm", "transformers", "huggingface_hub", "safetensors") if m in sys.modules)
motif3 = sorted(m for m in sys.modules if m.startswith("models.demos.motif3."))
print(json.dumps({"demos": demos, "other_models": other_models, "heavy": heavy, "motif3": motif3}))
"""


def test_import_is_self_contained():
    """README §13 import rule in a fresh interpreter (the probe of tests/unit/test_infra_import.py, whose MODULES list
    is shared infra; requested there): the kernels package and this module import only ttnn / torch."""
    import subprocess
    import sys

    tt_metal = Path(__file__).resolve().parents[5]
    mods = ["models.demos.motif3.tt.kernels", "models.demos.motif3.tt.kernels.sinkhorn_motif"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tt_metal) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, json.dumps(mods)],
        cwd=str(tt_metal),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, res.stderr[-4000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["demos"] == [] and out["other_models"] == [], out
    assert out["heavy"] == [], out
    assert set(out["motif3"]) == {"models.demos.motif3.tt", *mods}, out["motif3"]
    for marker in ("Opening user mode device driver", "Starting devices in cluster"):
        assert marker not in res.stderr and marker not in res.stdout


# =====================================================================================================================
# device tests
# =====================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_layout_passthrough(mesh_device, device_params):
    """mode "passthrough": outputs = the raw mixes columns, bit exact (transposes, DST layout, pack path)."""
    SM = _sm()
    _log_fabric(mesh_device, "sinkhorn_passthrough")
    consts = SM.consts_to_device(_make_consts(), mesh_device)
    for T in (8, 32, 100, 4096):
        g = torch.Generator().manual_seed(T)
        p = torch.randn(T, 32, generator=g) * torch.exp(4 * torch.randn(T, 32, generator=g))
        p[0, :24] = torch.tensor([1e-30, -1e-30, 3.4e38, -3.4e38, 1.0 + 2**-23, -(1.0 + 2**-23)] * 4)
        mix = _to_mesh(p, mesh_device)
        pre, post, H = SM.motif_sinkhorn(mix, consts, mode="passthrough")
        a, b, c = _read(pre), _read(post), _read(H)
        assert tuple(a.shape) == (T, 4) and tuple(b.shape) == (T, 4) and tuple(c.shape) == (T, 16)
        assert torch.equal(_bits(a), _bits(p[:, 0:4])), f"T={T}: h_pre layout"
        assert torch.equal(_bits(b), _bits(p[:, 4:8])), f"T={T}: h_post layout"
        assert torch.equal(_bits(c), _bits(p[:, 8:24])), f"T={T}: H layout"
        if T in (8, 4096):
            assert all(_replicas_identical(t) for t in (pre, post, H))
        print(f"[sinkhorn] passthrough T={T}: bit exact")
        _free([mix, pre, post, H])


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_logits_exact(mesh_device, device_params):
    """mode "logits": clamp(alpha p + b) (fp32 mul then add) bitwise equal to torch: zero differing elements (a
    compiler that re-fuses the multiply-add into one SFPMAD, the bug the DST round trip in logit() avoids, would differ
    by an ulp here and there)."""
    SM = _sm()
    _log_fabric(mesh_device, "sinkhorn_logits")
    cases = [(f"synthetic wide T={T}", _make_p(T, "wide", 9), _make_consts(7)) for T in (32, 4096)]
    cap = _capture_or_none()
    if cap is not None:
        for name in ("29.mhc_ffn", "4.mhc_attn"):
            cases.append((f"real {name} T=4096", cap["sites"][name]["p"], _site_consts(cap["sites"][name])))
    for tag, p, c in cases:
        consts = SM.consts_to_device(c, mesh_device)
        mix, outs, (lp, lq, lr) = _run(mesh_device, p, consts, mode="logits")
        g_pre, g_post, g_res = SM.motif_logits_torch(p, c)
        g = torch.cat([g_pre.clamp(-10, 10), g_post.clamp(-10, 10), g_res.reshape(-1, 16).clamp(-20, 20)], -1)
        d = torch.cat([lp, lq, lr], -1)
        ulp = (_bits(d).long() - _bits(g).long()).abs()
        n_diff = int((ulp > 0).sum())
        print(f"[sinkhorn] logits {tag}: {n_diff}/{ulp.numel()} elements differ, max {int(ulp.max())} ulp")
        assert n_diff == 0, f"{tag}: {n_diff} logits differ from torch (max {int(ulp.max())} ulp)"
        _free([mix, outs, consts])


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_synthetic_regimes(mesh_device, device_params):
    """Full mode vs the fp32 Motif golden on the G3 regimes (incl. the 1e-8 floor and +90 overflow)."""
    SM = _sm()
    _log_fabric(mesh_device, "sinkhorn_regimes")
    c = _make_consts(11)
    consts = SM.consts_to_device(c, mesh_device)
    worst = {}
    for T in (8, 32, 4096):
        for regime in ("realistic", "moderate", "wide", "peaked", "degenerate", "overflow"):
            p = _make_p(T, regime, 1000 * T + len(regime))
            mix, outs, dev = _run(mesh_device, p, consts)
            r = _compare(p, c, dev)
            ok_rep = _replicas_identical(outs[2]) if T == 32 else True
            print(
                f"[sinkhorn] regime {regime:10s} T={T:4d}: max|dH| {r['H']:.2e} (vs fp64 {r['H_vs_fp64']:.2e}; "
                f"golden32 vs fp64 {r['golden32_vs_fp64']:.2e}) |dh_pre| {r['h_pre']:.2e} |dh_post| {r['h_post']:.2e} "
                f"colsum-1 {r['colsum_dev']:.1e} replicas_identical={ok_rep}"
            )
            worst[(T, regime)] = r
            assert ok_rep
            _free([mix, outs])
    assert max(r["H"] for r in worst.values()) <= H_TOL
    assert max(max(r["h_pre"], r["h_post"]) for r in worst.values()) <= HPP_TOL


def _extreme_tokens(p: torch.Tensor, consts: torch.Tensor, k: int) -> torch.Tensor:
    l_pre, l_post, l_res = _sm().motif_logits_torch(p, consts)
    score = torch.maximum(
        l_res.reshape(-1, 16).abs().amax(-1) / 20.0, torch.cat([l_pre, l_post], -1).abs().amax(-1) / 10.0
    )
    return torch.topk(score, k).indices.sort().values


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_real_mixes(mesh_device, device_params):
    """Real mHC mixes (layers 0-5, 28-35, both sites) vs the reference's exact maps: max|dH| <= 1e-5 (self-imposed)."""
    SM = _sm()
    cap = _load_capture()
    _log_fabric(mesh_device, "sinkhorn_real")
    print(f"[sinkhorn] capture meta: {cap['meta']}")
    rows, worst = [], dict(H=0.0, h_pre=0.0, h_post=0.0)
    for name, site in cap["sites"].items():
        c = _site_consts(site)
        p = site["p"]
        # the golden equals the reference module's own taps (same fp32 ops, bitwise)
        g = SM.motif_mhc_maps_torch(p, c)
        tap_ok = (
            torch.equal(g[0], site["ref_h_pre"])
            and torch.equal(g[1], site["ref_h_post"])
            and torch.equal(g[2], site["ref_h_res"])
        )
        l_pre, l_post, l_res = SM.motif_logits_torch(p, c)
        consts = SM.consts_to_device(c, mesh_device)
        for T in (8, 32, N_TOKENS):
            idx = torch.arange(p.shape[0]) if T >= p.shape[0] else _extreme_tokens(p, c, T)
            pt = p[idx]
            mix, outs, dev = _run(mesh_device, pt, consts)
            r = _compare(pt, c, dev)
            rep = _replicas_identical(outs[2]) if T == 8 else True
            rows.append(
                dict(
                    site=name,
                    T=T,
                    **{k: r[k] for k in ("H", "h_pre", "h_post", "H_vs_fp64")},
                    max_abs_L_res=float(l_res[idx].abs().max()),
                    max_abs_L_pp=float(torch.cat([l_pre[idx], l_post[idx]], -1).abs().max()),
                    tap_ok=tap_ok,
                    replicas=rep,
                )
            )
            for k in worst:
                worst[k] = max(worst[k], r[k])
            assert rep, f"{name} T={T}: replicas differ"
            _free([mix, outs])
        _free(consts)
        tops = [x for x in rows if x["site"] == name]
        print(
            f"[sinkhorn] real {name:12s} |L_res|max {tops[-1]['max_abs_L_res']:5.1f} |L_pp|max {tops[-1]['max_abs_L_pp']:5.1f} "
            + " ".join(f"T={x['T']}: dH {x['H']:.1e} dpre {x['h_pre']:.1e} dpost {x['h_post']:.1e};" for x in tops)
            + f" golden==ref taps: {tap_ok}"
        )
    print(f"[sinkhorn] real worst over {len(rows)} cases: {worst}")
    assert all(x["tap_ok"] for x in rows), "golden differs from the reference module taps"
    assert worst["H"] <= H_TOL, worst
    assert max(worst["h_pre"], worst["h_post"]) <= HPP_TOL, worst


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_row_varying_decode(mesh_device, device_params):
    """README §12 decode setup: 32 lanes, lane l on DP row l // 8, each row's 8 lanes different, replicated over the
    TP chips of the row. Every chip vs the golden of its own row's lanes; TP replicas bitwise identical; DP rows differ.
    Stock and wrnc layouts, eager; mhc.py's [1, 8, 32] view of a tile whose padding rows / unused columns hold
    NaN / Inf; one trace (both layouts) replayed with new per-row lanes copied into the persistent input."""
    SM = _sm()
    from models.demos.motif3.tt import ccl as CCL
    from models.demos.motif3.tt import rope as RO
    from models.demos.motif3.tt.model_config import MotifTTConfig

    _log_fabric(mesh_device, "sinkhorn_rows")
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)
    R, C = (int(s) for s in tuple(mesh_device.shape))
    L = cfg.lanes_per_row
    assert cfg.max_batch == cfg.dp * L == 32
    ((name, p_all, c),) = _sites(["29.mhc_ffn"])
    consts = SM.consts_to_device(c, mesh_device)
    gen = torch.Generator().manual_seed(1)
    print(f"[sinkhorn] rows: site {name}, mesh {R}x{C}, dp={cfg.dp} lanes/row={L}, axes {cfg.axes}")

    def new_lanes():
        idx = torch.randperm(p_all.shape[0], generator=gen)[: cfg.max_batch]
        return p_all[idx]  # [32, 24] in lane order 8 dp + l

    def upload(p, device, garbage=False):
        rows = RO.lanes_to_rows(_pad32(p), cfg)  # [dp, 8, 32]
        if garbage:  # mhc.py decode view: the logical 8 rows of a [32, 32] tile, junk everywhere else
            junk = torch.tensor([float("nan"), float("inf"), -float("inf"), 3.0e38])
            full = junk[torch.arange(cfg.dp * TILE_ROWS * 32) % 4].reshape(cfg.dp, TILE_ROWS, 32).clone()
            full[:, :L, :] = rows
            full[:, :L, 24:] = junk[torch.arange(L * 8) % 4].reshape(L, 8)
            base = RO.shard_lanes(full, cfg, mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            return ttnn.reshape(base, ttnn.Shape([1, L, 32]), ttnn.Shape([1, TILE_ROWS, 32])), base  # (view, owner)
        return RO.shard_lanes(rows, cfg, mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def check(p, outs, layout, tag):
        for o in outs:
            assert CCL.replicas_identical(o, mesh_device, "tp"), f"{tag}: TP replicas differ"
        stacked = [CCL.device_tensors_to_torch(o, mesh_device) for o in outs]  # [R, C, *local]
        worst = 0.0
        for r in range(R):
            for cc in range(C):
                dp, tp = cfg.axes.roles(r, cc)
                pl = p[L * dp : L * dp + L]
                if layout == "stock":
                    e = _compare(pl, c, [t[r, cc] for t in stacked])
                    worst = max(worst, e["H"], e["h_pre"], e["h_post"])
                else:
                    e = _compare_wrnc(pl, c, stacked[0][r, cc], stacked[1][r, cc])
                    worst = max(worst, e["w_pre"], e["w_post"])
        H_rows = stacked[-1][:, 0]  # [R, ...] the TP-chip-0 copy of each DP row
        distinct = len({_bits(H_rows[r].contiguous()).numpy().tobytes() for r in range(R)})
        print(
            f"[sinkhorn] rows {tag}: worst max-abs over all 32 chips {worst:.2e}; TP replicas identical; "
            f"{distinct}/{R} distinct DP rows"
        )
        assert worst <= H_TOL, (tag, worst)
        assert distinct == R, f"{tag}: DP rows are not distinct (the per-row placement is not exercised)"

    p = new_lanes()
    mix = upload(p, mesh_device)
    print(f"[sinkhorn] rows: per-chip mixes {tuple(mix.shape)} padded {tuple(mix.padded_shape)}")
    for layout in ("stock", "wrnc"):
        outs = SM.motif_sinkhorn(mix, consts, layout=layout)
        check(p, outs, layout, f"eager {layout}")
        _free(outs)
        view, base = upload(p, mesh_device, garbage=True)
        outs = SM.motif_sinkhorn(view, consts, layout=layout)
        check(p, outs, layout, f"eager {layout} garbage-padded view {tuple(view.shape)}/{tuple(view.padded_shape)}")
        _free([outs, base])
    ttnn.synchronize_device(mesh_device)
    with _Capture(mesh_device) as cap:
        o_stock = SM.motif_sinkhorn(mix, consts)
        o_wrnc = SM.motif_sinkhorn(mix, consts, layout="wrnc")
    try:
        for rep in range(3):
            if rep:
                p = new_lanes()
                ttnn.copy_host_to_device_tensor(upload(p, None), mix)
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            check(p, o_stock, "stock", f"trace replay {rep} stock")
            check(p, o_wrnc, "wrnc", f"trace replay {rep} wrnc")
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    _free([mix, o_stock, o_wrnc, consts])


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_latency(mesh_device, device_params):
    """Eager and traced per-call latency vs the stock mhc_split_sinkhorn (G3: 52 us slope / 54 us raw traced)."""
    SM = _sm()
    _log_fabric(mesh_device, "sinkhorn_latency")
    c = _make_consts(11)
    consts = SM.consts_to_device(c, mesh_device)
    from models.demos.motif3.tests.unit.gates import goldens as gd

    stock_consts = _to_mesh(gd.build_sinkhorn_consts(4, (1.0, 1.0, 1.0), torch.zeros(24)), mesh_device)
    res = {}
    for T in (8, 32, 4096):
        p = _make_p(T, "wide", 3)
        mix = _to_mesh(_pad32(p), mesh_device)
        mix24 = _to_mesh(SM.motif_logits_torch(p, c)[0].new_zeros(T, 24), mesh_device)
        fixed = SM.allocate_outputs(mix)
        fns = {
            "motif": lambda: SM.motif_sinkhorn(mix, consts),
            "motif_prealloc": lambda: (SM.motif_sinkhorn(mix, consts, outputs=fixed), None)[1],
            "motif_refine": lambda: SM.motif_sinkhorn(mix, consts, div_refine=True),
            "stock": lambda: ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(mix24, stock_consts, 4, 20, 0.0),
        }
        if T <= 16:
            fns["motif_2halves"] = lambda: SM.motif_sinkhorn(mix, consts, num_halves=2)
        for name, fn in fns.items():
            eager = _eager_us(mesh_device, fn, iters=20 if T <= 32 else 5)
            traced, raw = _traced_us(mesh_device, fn, n=256 if T <= 32 else 64, reps=9)
            res[(name, T)] = (eager, traced, raw)
            print(
                f"[sinkhorn] latency {name:14s} T={T:4d}: eager {eager:8.1f} us/call, traced raw {raw:6.2f} us/call "
                f"(slope {traced:6.2f})"
            )
        # host-side cost per call (enqueue only, no synchronize): Python descriptor (cached) + generic_op dispatch
        for name in ("motif", "motif_prealloc", "stock"):
            fn = fns[name]
            _free(fn())
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            outs = [fn() for _ in range(20)]
            t_host = (time.perf_counter() - t0) / 20 * 1e6
            ttnn.synchronize_device(mesh_device)
            _free(outs)
            print(f"[sinkhorn] host enqueue {name:14s} T={T:4d}: {t_host:7.1f} us/call")
        _free([mix, mix24, fixed])
    for T in (8, 32):
        m, s = res[("motif", T)][2], res[("stock", T)][2]
        print(f"[sinkhorn] T={T}: traced raw {m:.2f} us vs stock {s:.2f} us -> {s / m:.1f}x")
        assert m < s, (T, res[("motif", T)], res[("stock", T)])


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_prefill_latency(mesh_device, device_params):
    """Stock vs wrnc layout at the prefill buckets (preallocated outputs): wrnc writes 24 fp32 tiles per 32 tokens
    (8x the stock layout's 3), so it is DRAM-write bound at large T; this is the cost tt/mhc.py weighs for prefill."""
    SM = _sm()
    _log_fabric(mesh_device, "sinkhorn_prefill_latency")
    c = _make_consts(23)
    consts = SM.consts_to_device(c, mesh_device)
    res = {}
    for T in PREFILL_BUCKETS:
        mix = _to_mesh(_pad32(_make_p(T, "wide", 70 + T)), mesh_device)
        for layout in ("stock", "wrnc"):
            outs = SM.allocate_outputs(mix, layout=layout)
            fn = lambda: (SM.motif_sinkhorn(mix, consts, outputs=outs, layout=layout), None)[1]  # noqa: E731
            eager = _eager_us(mesh_device, fn, iters=10, warmup=3)
            slope, raw = _traced_us(mesh_device, fn, n=16 if T <= 8192 else 8, reps=7)
            mb = sum(int(o.buffer_num_pages()) for o in outs) * SM.TILE_BYTES_FP32 / 1e6
            res[(layout, T)] = (eager, slope, raw)
            print(
                f"[sinkhorn] prefill latency {layout:5s} T={T:5d}: eager (prealloc, warmed) {eager:7.1f} us/call; "
                f"traced raw {raw:6.1f} us/call (slope {slope:6.1f}); outputs {mb:6.1f} MB/chip "
                f"({mb * 1e3 / max(raw, 1e-9):5.0f} MB/ms written)"
            )
            _free(outs)
        _free(mix)
    T = PREFILL_BUCKETS[-1]
    sites = 106
    print(
        f"[sinkhorn] prefill T={T}: {sites} sites x raw -> stock {sites * res[('stock', T)][2] / 1e3:.1f} ms, "
        f"wrnc {sites * res[('wrnc', T)][2] / 1e3:.1f} ms per prefill"
    )


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_token_counts_and_source(mesh_device, device_params):
    """Partial / multi-tile token counts up to the 32768 prefill bucket (both num_halves paths, uneven core split),
    wrnc at prefill buckets, MotifSinkhorn.from_source (with / without cfg) and the input / output formats."""
    SM = _sm()
    from models.demos.motif3.tt import weights as W
    from models.demos.motif3.tt.model_config import MotifTTConfig

    _log_fabric(mesh_device, "sinkhorn_token_counts")
    c = _make_consts(5)
    s = SM.scalars_from_consts(c)
    prefix = W.hf_name(7, "mhc_ffn")
    src = W.DictWeightSource({f"{prefix}.{k}": v.to(torch.bfloat16).float() for k, v in s.items()})
    site = SM.MotifSinkhorn.from_source(mesh_device, src, 7, "mhc_ffn")
    assert torch.equal(site.consts_host, SM.consts_from_scalars(W.mhc_scalars(src, prefix)))
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)
    site_cfg = SM.MotifSinkhorn.from_source(mesh_device, src, 7, "mhc_ffn", cfg=cfg)  # weights.as_tensor upload
    assert site_cfg.iters == cfg.sinkhorn_iters and float(cfg.mhc_h_post_coeff) == SM.H_POST_COEFF
    worst = 0.0
    for T in (1, 5, 16, 17, 31, 33, 64, 100, 1000, 3872, 8192, 16384, 32768):
        p = _make_p(T, "peaked", 77 + T)
        mix = _to_mesh(_pad32(p), mesh_device)
        outs = (site_cfg if T in (17, 8192) else site)(mix)
        dev = tuple(_read(o) for o in outs)
        assert tuple(dev[2].shape) == (T, 16)
        r = _compare(p, site.consts_host, dev)
        worst = max(worst, r["H"], r["h_pre"], r["h_post"])
        msg = f"[sinkhorn] T={T:5d}: max|dH| {r['H']:.2e} |dh_pre| {r['h_pre']:.2e} |dh_post| {r['h_post']:.2e}"
        if T >= 8192:
            rep = _replicas_identical(outs[2])
            assert rep, f"T={T}: replicas differ"
            msg += f" replicas_identical={rep}"
        if T in (8192, 32768):  # wrnc at prefill buckets (8x the output bytes; see test_prefill_latency)
            w = site(mix, layout="wrnc")
            e = _compare_wrnc(p, site.consts_host, _read(w[0]), _read(w[1]))
            worst = max(worst, e["w_pre"], e["w_post"])
            msg += f"; wrnc max|dw_pre| {e['w_pre']:.2e} max|dw_post| {e['w_post']:.2e}"
            _free(w)
        print(msg)
        _free([mix, outs])
    # input / output formats tt/mhc.py may use: 4D [1, 1, T, 32] (the MHC-2 projection output), logical width 24,
    # L1-interleaved input and outputs; reuse_outputs
    site_r = SM.MotifSinkhorn(mesh_device, site.consts_host, reuse_outputs=True)
    for T, shape_kind, mem in ((8, "4d", "dram"), (8, "w24", "l1"), (32, "4d", "l1"), (300, "w24", "dram")):
        p = _make_p(T, "wide", 3 * T)
        host = _pad32(p) if shape_kind == "4d" else p
        host = host.reshape(1, 1, T, -1) if shape_kind == "4d" else host
        mc = ttnn.L1_MEMORY_CONFIG if mem == "l1" else ttnn.DRAM_MEMORY_CONFIG
        mix = _to_mesh(host, mesh_device, memory_config=mc)
        for rep in range(2):
            outs = site_r(mix, memory_config=mc)
            dev = tuple(_read(o) for o in outs)
            r = _compare(p, site.consts_host, dev)
            worst = max(worst, r["H"], r["h_pre"], r["h_post"])
        assert outs[0].memory_config().buffer_type == mc.buffer_type
        print(
            f"[sinkhorn] format {shape_kind} {mem} T={T}: input {tuple(mix.shape)}, outputs "
            f"{[tuple(o.shape) for o in outs]} in {mem}: max|dH| {r['H']:.2e} (reused outputs)"
        )
        _free(mix)
    site_r.release()
    site.release()
    site_cfg.release()
    assert worst <= H_TOL


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_input_views_and_validation(mesh_device, device_params):
    """Views and padding: NaN / Inf in padding rows and unused columns do not leak into the logical rows; a view whose
    padded height exceeds ceil(T/32) tiles is processed as ceil(T/32) tiles (the outputs' size). Validation: every
    invalid input / output spec raises before a launch, also for a tensor placed at a recycled buffer address (a
    descriptor-cache hit by address). Descriptor-cache separation of interleaved layouts / modes / options."""
    SM = _sm()
    _log_fabric(mesh_device, "sinkhorn_views_validation")
    ((name, p0, c),) = _sites(["29.mhc_ffn"], n_synthetic=2048)
    consts = SM.consts_to_device(c, mesh_device)
    junk = torch.tensor([float("nan"), float("inf"), -float("inf"), 3.0e38])
    worst = 0.0

    def check(p, outs, layout, tag):
        nonlocal worst
        if layout == "stock":
            e = _compare(p, c, tuple(_read(o) for o in outs))
            w = max(e["H"], e["h_pre"], e["h_post"])
        else:
            e = _compare_wrnc(p, c, _read(outs[0]), _read(outs[1]))
            w = max(e["w_pre"], e["w_post"])
        worst = max(worst, w)
        print(f"[sinkhorn] {tag} {layout}: max-abs {w:.2e}")
        assert w <= H_TOL, (tag, layout, e)

    # ---- garbage-padded views (mhc.py decode view path) and over-padded views (n_tiles from the logical T) ----
    for T, rows in ((8, 32), (16, 32), (17, 32), (40, 64), (8, 64), (40, 96)):
        p = p0[torch.topk(p0[:, 8:].abs().amax(-1), T).indices]  # the site's most extreme tokens
        full = junk[torch.arange(rows * 32) % 4].reshape(rows, 32).clone()
        full[:T, :24] = p
        base = _to_mesh(full.reshape(1, 1, rows, 32), mesh_device)
        try:
            view = ttnn.reshape(base, ttnn.Shape([1, 1, T, 32]), ttnn.Shape([1, 1, rows, 32]))
        except Exception as e:  # noqa: BLE001
            print(f"[sinkhorn] view T={T} in {rows} rows: ttnn.reshape refused ({type(e).__name__}); skipped")
            _free(base)
            continue
        assert tuple(view.shape) == (1, 1, T, 32) and tuple(view.padded_shape) == (1, 1, rows, 32)
        n_tiles = -(-T // 32)
        for layout in ("stock", "wrnc"):
            outs = SM.allocate_outputs(view, layout=layout)
            desc = SM.program_descriptor(view, consts, *outs, layout=layout)
            assert list(desc.kernels[0].common_runtime_args)[2] == n_tiles  # reader: tiles of logical rows only
            assert list(desc.kernels[1].common_runtime_args)[len(outs)] == n_tiles  # writer: same page count
            _free(outs)
            outs = SM.motif_sinkhorn(view, consts, layout=layout)
            check(p, outs, layout, f"view T={T} in {rows} padded rows (NaN/Inf padding; {n_tiles} tile(s))")
            if layout == "stock":
                assert _replicas_identical(outs[2])
            _free(outs)
        _free(base)

    # ---- validation: invalid inputs / outputs raise before anything is launched ----
    T = 8
    p = p0[:T]
    mix = _to_mesh(_pad32(p), mesh_device)
    good = SM.allocate_outputs(mix)
    good_w = SM.allocate_outputs(mix, layout="wrnc")
    check(p, SM.motif_sinkhorn(mix, consts, outputs=good), "stock", "valid preallocated outputs")
    alloc = lambda shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG: (  # noqa: E731
        ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, mesh_device, mc)
    )
    shard_mc = ttnn.create_sharded_memory_config(
        (32, 32), ttnn.CoreGrid(y=1, x=1), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )
    bad_inputs = {
        "bf16 mixes": lambda: _to_mesh(_pad32(p), mesh_device, dtype=ttnn.bfloat16),
        "row-major mixes": lambda: _to_mesh(_pad32(p), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT),
        "width-16 mixes": lambda: _to_mesh(p[:, :16], mesh_device),
        "leading dim 2": lambda: _to_mesh(_pad32(p).reshape(1, T, 32).repeat(2, 1, 1), mesh_device),
        "sharded mixes": lambda: _to_mesh(_pad32(p), mesh_device, memory_config=shard_mc),
    }
    bad_outputs = {
        "2 outputs for stock": lambda: good[:2],
        "stock outputs for wrnc": lambda: ("wrnc", good[:2]),
        "wrnc outputs for stock": lambda: good_w,
        "bf16 output": lambda: (alloc([T, 4], ttnn.bfloat16), good[1], good[2]),
        "output for T+1": lambda: (good[0], good[1], alloc([T + 1, 16])),
        "rank-4 output": lambda: (alloc([1, 1, T, 4]), good[1], good[2]),
        "row-major output": lambda: (good[0], alloc([T, 4], layout=ttnn.ROW_MAJOR_LAYOUT), good[2]),
        "sharded output": lambda: (alloc([T, 4], mc=shard_mc), good[1], good[2]),
        "host output": lambda: (
            ttnn.from_torch(torch.zeros(T, 4), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
            good[1],
            good[2],
        ),
        "torch output": lambda: (torch.zeros(T, 4), good[1], good[2]),
    }
    n_raised = 0
    for tag, make in bad_inputs.items():
        bad = make()
        with pytest.raises(ValueError):
            SM.motif_sinkhorn(bad, consts)
        n_raised += 1
        _free(bad)
    with pytest.raises(ValueError):
        SM.motif_sinkhorn(mix, _to_mesh(torch.zeros(32, 32), mesh_device))  # one-tile consts
    n_raised += 1
    for tag, make in bad_outputs.items():
        made = make()
        layout, outs = made if isinstance(made, tuple) and isinstance(made[0], str) else ("stock", made)
        with pytest.raises((ValueError, TypeError)):
            SM.motif_sinkhorn(mix, consts, outputs=outs, layout=layout)
        n_raised += 1
        _free([o for o in outs if all(o is not g for g in (*good, *good_w))])
    # recycled address: free a validated output and put a bf16 tensor of another shape in its place -> the cache is
    # keyed by the address, but the spec check sends it back through the validation
    addr = good[2].buffer_address()
    ttnn.deallocate(good[2])
    impostor = alloc([T, 16], ttnn.bfloat16)
    same = impostor.buffer_address() == addr
    with pytest.raises(ValueError):
        SM.motif_sinkhorn(mix, consts, outputs=(good[0], good[1], impostor))
    n_raised += 1
    _free(impostor)
    good = (good[0], good[1], alloc([T, 16]))
    check(p, SM.motif_sinkhorn(mix, consts, outputs=good), "stock", "valid outputs again after the rejections")
    # the same for the input: a bf16 mixes tensor at a validated mixes tensor's address
    addr = mix.buffer_address()
    _free(mix)
    imp_mix = _to_mesh(_pad32(p), mesh_device, dtype=ttnn.bfloat16)
    same_mix = imp_mix.buffer_address() == addr
    with pytest.raises(ValueError):
        SM.motif_sinkhorn(imp_mix, consts, outputs=good)
    n_raised += 1
    _free(imp_mix)
    print(
        f"[sinkhorn] validation: {n_raised} invalid calls raised; recycled-address cases exercised a cache hit by "
        f"address: output {same}, mixes {same_mix}"
    )
    _free([good, good_w])

    # ---- descriptor-cache separation: layouts / modes / options interleaved on the same tensors ----
    p = p0[:1000]
    mix = _to_mesh(_pad32(p), mesh_device)
    consts_l1 = SM.consts_to_device(c, mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
    for rnd in range(2):
        check(p, SM.motif_sinkhorn(mix, consts), "stock", f"interleaved round {rnd}")
        check(p, SM.motif_sinkhorn(mix, consts, layout="wrnc"), "wrnc", f"interleaved round {rnd}")
        lp, lq, lr = (_read(o) for o in SM.motif_sinkhorn(mix, consts, mode="logits"))
        g = SM.motif_logits_torch(p, c)
        assert torch.equal(lp, g[0].clamp(-10, 10)) and torch.equal(lq, g[1].clamp(-10, 10))
        assert torch.equal(lr, g[2].reshape(-1, 16).clamp(-20, 20))
        a, _, h = (_read(o) for o in SM.motif_sinkhorn(mix, consts, mode="passthrough"))
        assert torch.equal(_bits(h), _bits(_pad32(p)[:, 8:24])) and torch.equal(_bits(a), _bits(_pad32(p)[:, :4]))
    check(p, SM.motif_sinkhorn(mix, consts, div_refine=True), "stock", "div_refine")
    check(p, SM.motif_sinkhorn(mix, consts, max_cores=7), "stock", "max_cores=7 (uneven split)")
    check(p, SM.motif_sinkhorn(mix, consts_l1), "stock", "L1 consts")
    for iters in (0, 1, 5, 40):
        outs = SM.motif_sinkhorn(mix, consts, iters=iters)
        g = SM.motif_mhc_maps_torch(p, c, iters=iters)
        d = _read(outs[2])
        dH = _maxabs(g[2], d)
        rel = float(((g[2].double() - d.double()).abs() / g[2].double().abs().clamp(min=1e-30)).max())
        print(f"[sinkhorn] iters={iters}: max|dH| {dH:.2e} (max rel {rel:.2e})")
        assert (dH <= H_TOL) if iters > 0 else (rel <= 1e-5)  # iters 0: raw exp(clamp(L)) up to 4.9e8 -> relative
    _free([mix, consts_l1, consts])
    print(f"[sinkhorn] views / validation worst max-abs {worst:.2e}")


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_trace_replay(mesh_device, device_params):
    """Decode usage: precompile (prepare_generic_op), capture once, replay with new inputs copied into the persistent
    mixes tensor; every replay must equal the golden of its inputs; replicas bitwise identical. Then a model-like
    3-site trace (different consts and inputs, stock and wrnc layouts mixed) replayed with new inputs."""
    SM = _sm()
    _log_fabric(mesh_device, "sinkhorn_trace")
    c = _make_consts(13)
    consts = SM.consts_to_device(c, mesh_device)
    for T in (8, 32):
        ps = [_make_p(T, regime, 500 + T + i) for i, regime in enumerate(("wide", "peaked", "degenerate"))]
        mix = _to_mesh(_pad32(ps[0]), mesh_device)
        outs0 = SM.allocate_outputs(mix)
        ttnn.experimental.prepare_generic_op([mix, consts, *outs0], SM.program_descriptor(mix, consts, *outs0))
        _free(outs0)
        # prepare_generic_op compiles and caches the program, but the kernel binaries are written to DRAM lazily at
        # the first enqueue: one eager call must precede the capture ("Cannot load new binaries during trace capture")
        _free(SM.motif_sinkhorn(mix, consts))
        ttnn.synchronize_device(mesh_device)
        with _Capture(mesh_device) as cap:
            outs = SM.motif_sinkhorn(mix, consts)
        try:
            for i, p in enumerate(ps):
                if i:
                    host = ttnn.from_torch(
                        _pad32(p),
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    )
                    ttnn.copy_host_to_device_tensor(host, mix)
                ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                dev = tuple(_read(o) for o in outs)
                r = _compare(p, c, dev)
                rep = all(_replicas_identical(o) for o in outs)
                print(
                    f"[sinkhorn] trace T={T} replay {i}: max|dH| {r['H']:.2e} |dh_pre| {r['h_pre']:.2e} "
                    f"|dh_post| {r['h_post']:.2e} replicas_identical={rep}"
                )
                assert r["H"] <= H_TOL and max(r["h_pre"], r["h_post"]) <= HPP_TOL and rep
        finally:
            ttnn.release_trace(mesh_device, cap.tid)
        _free([mix, outs])
    _free(consts)

    # ---- 3 sites in one capture, layouts mixed (wrnc for the middle site), replayed with new inputs ----
    sites = _sites(["29.mhc_ffn", "28.mhc_attn", "0.mhc_ffn"])
    layouts = ("stock", "wrnc", "stock")
    s_consts = [SM.consts_to_device(cs, mesh_device) for _, _, cs in sites]
    T = 8
    gen = torch.Generator().manual_seed(5)
    inputs = [[pa[torch.randperm(pa.shape[0], generator=gen)[:T]] for _, pa, _ in sites] for _ in range(3)]
    mixes = [_to_mesh(_pad32(x), mesh_device) for x in inputs[0]]
    for m, cs, lay in zip(mixes, s_consts, layouts):  # eager warm-up of each (shape, layout) program
        _free(SM.motif_sinkhorn(m, cs, layout=lay))
    ttnn.synchronize_device(mesh_device)
    with _Capture(mesh_device) as cap:
        outs = [SM.motif_sinkhorn(m, cs, layout=lay) for m, cs, lay in zip(mixes, s_consts, layouts)]
    try:
        for rep in range(3):
            if rep:
                for m, x in zip(mixes, inputs[rep]):
                    host = ttnn.from_torch(
                        _pad32(x),
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    )
                    ttnn.copy_host_to_device_tensor(host, m)
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            msg = []
            for k, ((nm, _, cs), lay) in enumerate(zip(sites, layouts)):
                x = inputs[rep][k]
                if lay == "stock":
                    e = _compare(x, cs, tuple(_read(o) for o in outs[k]))
                    w = max(e["H"], e["h_pre"], e["h_post"])
                else:
                    e = _compare_wrnc(x, cs, _read(outs[k][0]), _read(outs[k][1]))
                    w = max(e["w_pre"], e["w_post"])
                assert w <= H_TOL, (rep, nm, lay, e)
                assert all(_replicas_identical(o) for o in outs[k]), (rep, nm, lay)
                msg.append(f"{nm} {lay} {w:.2e}")
            print(f"[sinkhorn] 3-site trace replay {rep}: " + "; ".join(msg) + "; replicas identical")
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(20):
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        print(
            f"[sinkhorn] 3-site trace replay: {(time.perf_counter() - t0) / 20 * 1e6:.1f} us per replay "
            f"(incl. 1/20 of a sync)"
        )
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    _free([mixes, outs, s_consts])


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_host_overhead_breakdown(mesh_device, device_params):
    """Host-side cost of one eager call, by component (32-chip mesh): output allocation, cache key + spec check,
    validation (descriptor-cache miss only), dispatch."""
    SM = _sm()
    _log_fabric(mesh_device, "sinkhorn_host_overhead")
    consts = SM.consts_to_device(_make_consts(), mesh_device)
    N = 50

    def per_call(fn):
        fn()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(N):
            fn()
        dt = (time.perf_counter() - t0) / N * 1e6
        ttnn.synchronize_device(mesh_device)
        return dt

    kw = dict(SM._DEFAULT_OPTIONS)
    for T in (8, 4096):
        mix = _to_mesh(_pad32(_make_p(T, "wide", 1)), mesh_device)
        fixed = SM.allocate_outputs(mix)
        spec = fixed[0].spec
        res = {}
        res["alloc3+free3 (shape API)"] = per_call(lambda: _free(SM.allocate_outputs(mix)))
        res["alloc1+free1 (spec API)"] = per_call(lambda: _free(ttnn.allocate_tensor_on_device(spec, mesh_device)))
        res["lookup key (options, mesh ids, addresses, specs)"] = per_call(
            lambda: SM._lookup_key(mix, consts, fixed, kw)
        )
        res["validate (miss path only)"] = per_call(lambda: SM._validate(mix, consts, fixed, "stock"))
        res["program_descriptor (miss path)"] = per_call(lambda: SM.program_descriptor(mix, consts, *fixed))
        res["cached_descriptor (hit)"] = per_call(lambda: SM._cached_descriptor(mix, consts, fixed))
        desc = SM._cached_descriptor(mix, consts, fixed)
        res["generic_op (prebuilt descriptor)"] = per_call(lambda: ttnn.generic_op([mix, consts, *fixed], desc))
        res["motif_sinkhorn(outputs=fixed)"] = per_call(lambda: SM.motif_sinkhorn(mix, consts, outputs=fixed))
        if T == 4096:
            for mc in (16, 32, 64):
                res[f"motif_sinkhorn(outputs=fixed, max_cores={mc})"] = per_call(
                    lambda: SM.motif_sinkhorn(mix, consts, outputs=fixed, max_cores=mc)
                )
        for k, v in res.items():
            print(f"[sinkhorn] host T={T:4d} {k:50s} {v:8.1f} us/call")
        _free([mix, fixed])


def _glue_to_wrnc(h_pre, h_post, H):
    """A representative ttnn glue from the stock layout ([T,4], [T,4], [T,16]) to the attn_res_weighted_reduce_nc
    weights ([1,4,T,1], [4,5,T,1]); what tt/mhc.py needs without layout="wrnc" (for the latency comparison)."""
    T = int(h_pre.shape[0])
    w_pre = ttnn.permute(ttnn.reshape(h_pre, [1, 1, T, 4]), (0, 3, 2, 1))  # [1, 4, T, 1]
    H3 = ttnn.reshape(H, [T, 4, 4])
    hp3 = ttnn.reshape(h_post, [T, 4, 1])
    w = ttnn.concat([H3, hp3], dim=-1)  # [T, 4, 5]
    w_post = ttnn.permute(ttnn.reshape(w, [1, T, 4, 5]), (2, 3, 1, 0))  # [4, 5, T, 1]
    return w_pre, w_post


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_wrnc_layout(mesh_device, device_params):
    """layout="wrnc": w_pre [1,4,T,1] / w_post [4,5,T,1] equal the stock outputs re-laid out (bitwise), feed
    attn_res_weighted_reduce_nc correctly (x_red and the fused post [X | out]), and save the glue ops."""
    SM = _sm()
    _log_fabric(mesh_device, "sinkhorn_wrnc")
    c = _make_consts(17)
    consts = SM.consts_to_device(c, mesh_device)
    wrnc = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc
    for T in (8, 32, 1024):
        p = _make_p(T, "wide", 900 + T)
        mix = _to_mesh(_pad32(p), mesh_device)
        stock = SM.motif_sinkhorn(mix, consts)
        w_pre, w_post = SM.motif_sinkhorn(mix, consts, layout="wrnc")
        assert tuple(w_pre.shape) == (1, 4, T, 1) and tuple(w_post.shape) == (4, 5, T, 1)
        e_pre, e_post = SM.wrnc_weights_torch(*(_read(o) for o in stock))
        d_pre, d_post = _read(w_pre), _read(w_post)
        assert torch.equal(_bits(d_pre), _bits(e_pre)) and torch.equal(_bits(d_post), _bits(e_post)), f"T={T}"
        g_pre, g_post = SM.wrnc_weights_torch(*SM.motif_mhc_maps_torch(p, c))
        e_w = max(_maxabs(g_pre, d_pre), _maxabs(g_post, d_post))
        # end to end through the stream-mix op (bf16 streams, fp32 weights; golden: fp32 einsum + one bf16 cast)
        g = torch.Generator().manual_seed(T)
        X = torch.randn(1, 4, T, 4096, generator=g).to(torch.bfloat16)
        out = torch.randn(1, 1, T, 4096, generator=g).to(torch.bfloat16)
        Xt = ttnn.from_torch(
            X,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        Xc = ttnn.from_torch(
            torch.cat([X, out], dim=1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        x_red = _read(wrnc(Xt, w_pre, dim=1))
        x_new = _read(wrnc(Xc, w_post, dim=1))  # [4, 1, T, 4096]
        gx_red = torch.einsum("ct,bctd->btd", g_pre[0, :, :, 0], X.float()).to(torch.bfloat16).float()
        gx_new = (
            torch.einsum("rct,bctd->rtd", g_post[..., 0], torch.cat([X, out], 1).float()).to(torch.bfloat16).float()
        )
        from models.common.utility_functions import comp_pcc

        _, pcc_red = comp_pcc(gx_red.reshape(-1), x_red.reshape(-1), 0.99999)
        _, pcc_new = comp_pcc(gx_new.reshape(-1), x_new.reshape(-1), 0.99999)
        print(
            f"[sinkhorn] wrnc T={T:4d}: layout bit exact vs stock outputs; max|dw| vs golden {e_w:.2e}; "
            f"x_red PCC {pcc_red:.7f} max|d| {_maxabs(gx_red, x_red):.3e}; "
            f"post PCC {pcc_new:.7f} max|d| {_maxabs(gx_new, x_new.reshape(gx_new.shape)):.3e}"
        )
        assert e_w <= H_TOL and pcc_red > 0.99999 and pcc_new > 0.99999
        if T <= 32:
            gw = _glue_to_wrnc(*stock)
            assert torch.equal(_bits(_read(gw[0])), _bits(e_pre)) and torch.equal(_bits(_read(gw[1])), _bits(e_post))
            t_wrnc, raw_w = _traced_us(mesh_device, lambda: SM.motif_sinkhorn(mix, consts, layout="wrnc"), n=256)
            t_stock, raw_s = _traced_us(mesh_device, lambda: SM.motif_sinkhorn(mix, consts), n=256)

            def stock_plus_glue():
                o = SM.motif_sinkhorn(mix, consts)
                gw = _glue_to_wrnc(*o)
                _free(o)
                return gw

            t_glue, raw_g = _traced_us(mesh_device, stock_plus_glue, n=64)  # 7 ops per call
            print(
                f"[sinkhorn] wrnc T={T:4d} traced raw: layout=wrnc {raw_w:.2f} us (slope {t_wrnc:.2f}); stock "
                f"{raw_s:.2f} (slope {t_stock:.2f}); stock + glue {raw_g:.2f} us (slope {t_glue:.2f})"
            )
            _free(gw)
        _free([mix, stock, w_pre, w_post, Xt, Xc])
