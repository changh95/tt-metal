# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of ``tt/mhc.py`` (``MHCSite``) on the BH Galaxy, mesh (4, 8). Run ONLY through the device lock:

    scripts/devrun.sh -t 1500 -n mhc -- python -m pytest models/demos/motif3/tests/unit/test_mhc.py -s -p no:cacheprovider

Golden: ``models.demos.motif3.reference.modules.MHCLayer`` built in fp64 ("fp32-faithful": fp64 projection on the
bf16 streams with fp32 weights, Motif's fp32 Sinkhorn, fp64 mixing), plus the bf16 model (HF cast points) for
information. Every test logs the committed fabric (``[motif3.fabric] ... committed:``) and checks that replicas are
bitwise identical (decode: along TP, each DP row has its own 8 lanes; prefill: all 32 chips).

Checks per site (``_ok``; thresholds below):

* end to end vs the fp64 reference: max|dH|, max|dh_pre|, max|dh_post| (MHC-5 for the default "motif" backend; a
  regression bound for the explicit "stock" backend), PCC of x_red / X' >= 0.99999, ``p`` relative error;
* the mixing alone (``mix_errors``): x_red / X' against the fp64 einsum of the device's *own* (unrounded) coefficients
  on the same bf16 inputs, in bf16 ulps of the largest summand (G3's criterion: <= C = 4 / 5 means at most one extra
  rounding), and the signed relative bias against the *correctly rounded* exact result, i.e. HF's fp32 einsum + one
  bf16 cast (|bias| <= 1.5e-4 per site, <= 2.5e-5 over all real decode sites; the FPU's TF32 truncation of the fp32
  weights gave -3.6e-4; the bias of correct rounding itself is not counted);
* the coefficient maps alone (``maps_errors``): h_pre / h_post / H against the fp32 golden maps of the device's own raw
  mixes ``p`` (motif kernel: fp32-exact, <= 1e-5 / 1e-6), and how many logits the +-10 / +-20 clamps cut (the
  "peaked" regime must exercise both).

Tests:

* ``test_mhc_kernels`` -- the generic_op kernels of tt/mhc.py in isolation (``finalize_mixes`` decode and multi-core
  prefill layouts vs fp64, ``coefficient_layout`` bit exact incl. the TF32 rounding, ``post_mix`` == the stock op on
  the concat bitwise), the zero-copy views, the strict Sinkhorn backend selection and the mesh-keyed shared constants;
  run first, short.
* ``test_mhc_cache_roundtrip`` -- TT weight cache write + reload (no source access), bitwise-identical outputs, cache
  names.
* ``test_mhc_random`` -- random weights at real dims (``realistic`` and ``peaked`` logits beyond the +-10 / +-20 clamps),
  decode (per-row lanes) and prefill (S = 256 bucket, and a logical T = 200), both Sinkhorn backends ("motif" kernel,
  "stock" pre-clamped op), the ttnn glue, the stock post on the concat and the exact composite mixing path.
* ``test_mhc_real_decode`` -- real weights and real streams (MHC-5): layers 0-5 and 28-35, both sites, two prompts,
  32 tokens per site (16 hardest by the emulated TF32 Sinkhorn sensitivity + 16 random) spread over the 4 DP rows.
* ``test_mhc_real_prefill`` -- real weights/streams at prefill: default prompt (layers 0, 2, 4, both sites, S = 256)
  and the long prompt at the worst C3 site (layer 30 mhc_attn: S = 2048 real tokens, S = 4096 tiled).
* ``test_mhc_decode_trace_latency`` -- decode trace capture / replay with new inputs (trace safety), layout kernel vs
  ttnn glue, and the per-site latency breakdown, eager and traced, for the backend / layout / L1 / finalize / post
  variants.
* ``test_mhc_decode_fused`` -- Phase C D3: the fused decode site (``decode="fused"``) bitwise against the op path
  (random sites at T = 8 / 16 / 32, trace replay, the packed tile, real decode sites).
* ``test_mhc_prefill_latency`` -- prefill S = 128 ... 32768 (no chunking needed): warmed eager latency per stage,
  the kernel finalize / concat-free post against their op forms, and accuracy on 256 sampled tokens.
* ``test_mhc_probe_*`` -- opt-in (``MOTIF3_MHC_PROBE=1``) numerics / config probes behind the design choices.

Real-stream data (bf16, ~190 MB) lives under ``tt_cache/test/mhc/`` and is produced on the host (no device) by::

    scripts/hostrun.sh -t 3000 -- python -m models.demos.motif3.tests.unit.test_mhc --capture
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.motif3.tt import weights as W
from models.demos.motif3.tt.ccl import device_tensors_to_torch, log_fabric, replicas_identical
from models.demos.motif3.tt.model_config import DEFAULT_HF_META_DIR, MotifTTConfig, device_params

DATA = Path("/home/ttuser/hchang/experiments/motif-3/tt_cache/test/mhc")
HF_META = str(DEFAULT_HF_META_DIR)
TRACE = 128 * 1024 * 1024
MESH = [pytest.param((4, 8), device_params("FABRIC_2D_TORUS_XY", TRACE), id="4x8-torus2d")]
D = 4096
N = 4
REAL_DECODE_LAYERS = (0, 1, 2, 3, 4, 5, 28, 29, 30, 31, 32, 33, 34, 35)
BACKENDS = ("motif", "stock")  # both always run: a missing / broken kernel module fails the tests (no silent drop)

# end to end vs the fp64 reference (MHC-5 for the default "motif" backend)
TH_H, TH_PRE, TH_POST = 5e-3, 2e-3, 2e-3
# explicit "stock" backend (draft-1 pre-clamped op, TF32 arithmetic): regression bounds (measured real decode max|dH|
# 5.14e-3, h 9.5e-4)
TH_H_STOCK, TH_H_STOCK_PP = 6e-3, 2e-3
TH_PCC = 0.99999  # x_red / X' (measured >= 0.9999984 on real data); PCC ignores scale: see the bias bound
TH_P_REL = {"real": 1e-3, "random": 5e-3}  # ||p - p_ref|| / ||p_ref|| (measured 0.8-2.4e-4 real, 1.6-1.9e-3 random)
# the mixing alone vs the fp64 einsum of the device's own coefficients
TH_MIX_ULP = {"red": 4.0, "out": 5.0}  # bf16 ulps of the largest summand (G3: C = 4 / 5 = at most one extra rounding)
# |signed relative bias| per site and over all real decode sites. TF32-truncated weights gave -3.3e-4 everywhere;
# with nearest-even TF32 weights the per-site bias is noise: sigma ~2.5e-5 for a decode x_red (32 tokens x 4
# weights, each off by <= 2^-11 relative), so 1.5e-4 is 6 sigma there; the aggregate over 56 sites is the tight guard.
TH_MIX_BIAS, TH_MIX_BIAS_AGG = 1.5e-4, 2.5e-5
# the maps alone vs the fp32 golden maps of the device's own p
TH_MAPS_MOTIF = (1e-5, 1e-6)  # (H, h_pre / h_post): the kernel module's fp32-exactness targets (measured <= 3e-7)
TH_MAPS_STOCK = (6e-3, 2e-3)  # realistic / real logits (the stock op's TF32 arithmetic; G3 pre-clamped 0.8-1.4e-3)


# =====================================================================================================================
# helpers
# =====================================================================================================================
def _free(o):
    if isinstance(o, (list, tuple)):
        for x in o:
            _free(x)
    elif hasattr(o, "deallocate") and not isinstance(o, ttnn.Tensor):
        o.deallocate()
    elif isinstance(o, ttnn.Tensor):
        try:
            ttnn.deallocate(o)
        except Exception:
            pass


class _Capture:
    """Exception-safe trace capture (a dangling capture once hung close_mesh_device; GATES_RESULTS §11.6)."""

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


def _traced_us(mesh_device, fn, n=32, reps=7):
    """Gate slope method: (min t(n) - min t(n/2)) / (n/2) (sync alone ~140 us on this mesh; replays are bimodal)."""
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    t1 = _trace_min_us(mesh_device, fn, n // 2, reps)
    t2 = _trace_min_us(mesh_device, fn, n, reps)
    return max((t2 - t1) / (n - n // 2), 0.0)


def _eager_us(mesh_device, fn, iters=10):
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


def _pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    den = float(a.norm() * b.norm())
    return float((a @ b) / den) if den > 0 else (1.0 if torch.equal(a, b) else 0.0)


def _maxabs(a, b):
    d = (a.double() - b.double()).abs()
    return float(d.max()) if bool(torch.isfinite(d).all()) else float("inf")


def _bf16_ulp(x):
    x = x.double().abs()
    return torch.where(x == 0, torch.full_like(x, 2.0**-133), 2.0 ** (torch.floor(torch.log2(x)) - 7))


def _cfg(mesh_device):
    return MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device)


def _km():
    from models.demos.motif3.tt.kernels import sinkhorn_motif

    return sinkhorn_motif


def _per_row(mesh_device, cfg, host, dtype=ttnn.bfloat16):
    """host ``[dp, ...]`` -> chip (dp, tp) holds ``host[dp][None]`` (replicated over TP)."""
    return ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=W.mesh_mapper(mesh_device, cfg.axes, dp_dim=0),
    )


def _replicated(mesh_device, host, dtype=ttnn.bfloat16, memory_config=None):
    return ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _chips(t, mesh_device):
    """``[R, C, *local]`` float64 readback (logical shapes)."""
    return device_tensors_to_torch(t, mesh_device).double()


def _replicas_all(t, mesh_device, max_rows=256):
    """Bitwise identity of ``t`` on all 32 chips (prefill runs replicated everywhere), with one readback of every chip.
    Tensors with more than ``2 * max_rows`` rows are compared on their first and last ``max_rows`` rows (device row
    slices), which bounds the host traffic (a full X' at S = 32768 is 1.07 GB per chip)."""
    shape = [int(d) for d in t.shape]
    rows = shape[-2]
    windows = [None] if rows <= 2 * max_rows else [0, rows - max_rows]
    ok = True
    for a in windows:
        part = t
        if a is not None:
            start, end = [0] * len(shape), list(shape)
            start[-2], end[-2] = a, a + max_rows
            part = ttnn.slice(t, start, end)
        sh = device_tensors_to_torch(part, mesh_device)  # [R, C, ...]
        ok &= bool(torch.equal(sh, sh[0:1, 0:1].expand_as(sh)))
        if part is not t:
            ttnn.deallocate(part)
    return ok


# =====================================================================================================================
# weights and goldens (CPU)
# =====================================================================================================================
def site_tensors(source, prefix):
    """(proj [24, 16384], gamma [16384], scalars) of one site in fp32."""
    proj = W.mhc_projection_from_source(source, prefix).float()
    gamma = source.get(f"{prefix}.rms_norm.weight").float()
    return proj, gamma, W.mhc_scalars(source, prefix)


def ref_layer(proj, gamma, scalars, dtype=torch.float64):
    """``reference.modules.MHCLayer`` with the site's weights in ``dtype`` (fp64 = the fp32-faithful golden)."""
    from models.demos.motif3.reference.modules import MHCLayer

    m = MHCLayer(N, D, 20, 1.0, 1e-6, mix_fp32=False).to(dtype)
    with torch.no_grad():
        m.proj_merged.weight.copy_(proj.to(dtype))
        m.rms_norm.weight.copy_(gamma.to(dtype))
        for k in ("alpha_pre", "alpha_post", "alpha_res"):
            getattr(m, k).copy_(scalars[k].reshape(1).to(dtype))
        m.bias_pre.copy_(scalars["bias_pre"].reshape(4).to(dtype))
        m.bias_post.copy_(scalars["bias_post"].reshape(4).to(dtype))
        m.bias_res.copy_(scalars["bias_res"].reshape(4, 4).to(dtype))
    return m


@torch.no_grad()
def golden(m, x, out, dtype=torch.float64):
    """x ``[T, 4, D]`` (bf16 values), out ``[T, D]`` -> dict(p [T,24], h_pre [T,4], h_post [T,4], H [T,16],
    x_red [T, D], x_out [T, 4, D]) from the reference module ``m`` (fp64 or bf16)."""
    from models.demos.motif3.reference.golden import TensorRecorder
    from models.demos.motif3.reference.modules import MHCLayer

    xx = x.to(dtype)[None]
    rec = TensorRecorder()
    h_pre, h_post, h_res = m(xx, tap=rec)
    x_red = MHCLayer.pre(xx, h_pre)[0]
    x_out = MHCLayer.post(xx, out.to(dtype)[None], h_post, h_res)[0]
    T = x.shape[0]
    return dict(
        p=rec.tensors["mixes"][0].double(),
        h_pre=h_pre[0].double(),
        h_post=h_post[0].double(),
        H=h_res[0].double().reshape(T, 16),
        x_red=x_red.double(),
        x_out=x_out.double(),
    )


def random_site_source(layer, site, seed, regime="realistic"):
    """HF-named random weights of one site (reference ``random_state_dict`` scales, bf16-representable values).
    ``peaked``: alphas x40 / x20 and wider biases so that many logits pass the +-20 / +-10 clamps."""
    g = torch.Generator().manual_seed(seed)
    p = W.hf_name(layer, site)
    r = lambda t: t.to(torch.bfloat16).float()  # noqa: E731
    sd = {
        f"{p}.proj_pre.weight": r(torch.randn(4, 4 * D, generator=g) / 128),
        f"{p}.proj_post.weight": r(torch.randn(4, 4 * D, generator=g) / 128),
        f"{p}.proj_res.weight": r(torch.randn(16, 4 * D, generator=g) / 128),
        f"{p}.rms_norm.weight": r(torch.rand(4 * D, generator=g) + 0.5),
        f"{p}.bias_pre": r(0.3 * torch.randn(4, generator=g)),
        f"{p}.bias_post": r(0.3 * torch.randn(4, generator=g)),
        f"{p}.bias_res": r(0.3 * torch.randn(4, 4, generator=g)),
        f"{p}.alpha_pre": r(torch.rand(1, generator=g) * 0.3 + 0.3),
        f"{p}.alpha_post": r(torch.rand(1, generator=g) * 0.3 + 0.3),
        f"{p}.alpha_res": r(torch.rand(1, generator=g) * 0.3 + 0.3),
    }
    if regime == "peaked":
        sd[f"{p}.alpha_res"] = r(sd[f"{p}.alpha_res"] * 40)
        sd[f"{p}.alpha_pre"] = r(sd[f"{p}.alpha_pre"] * 20)
        sd[f"{p}.alpha_post"] = r(sd[f"{p}.alpha_post"] * 20)
        sd[f"{p}.bias_res"] = r(sd[f"{p}.bias_res"] * 10)
    return W.DictWeightSource(sd)


def random_streams(T, seed, scale=1.0):
    """[T, 4, D] bf16 streams (stream-correlated like real residuals) and out [T, D] bf16."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(T, 1, D, generator=g)
    x = base + 0.5 * torch.randn(T, 4, D, generator=g)
    x = x * torch.exp(0.25 * torch.randn(T, 1, 1, generator=g)) * scale
    out = torch.randn(T, D, generator=g) * 0.7
    return x.to(torch.bfloat16), out.to(torch.bfloat16)


def to_tt_layout(x):
    """reference [T, 4, D] -> TT stream-major [1, 4, T, D]."""
    return x.permute(1, 0, 2).unsqueeze(0).contiguous()


# =====================================================================================================================
# stage-isolating checks (CPU, fp64)
# =====================================================================================================================
def _rne_bf16(v):
    """fp64 -> the nearest bf16 value, ties to even, exactly (no fp64 -> fp32 -> bf16 double rounding); 0 stays 0."""
    a = v.abs()
    ulp = torch.pow(2.0, torch.floor(torch.log2(torch.where(a > 0, a, torch.ones_like(a)))) - 7)
    return torch.where(a > 0, torch.round(v / ulp) * ulp, v)


def mix_errors(xs, out, h_pre, h_post, H, x_red, x_out, chunk=256):
    """The mixing alone: device ``x_red [T, D]`` / ``X' [4, T, D]`` against the fp64 einsum of the device's own
    (unrounded) coefficients ``h_pre / h_post [T, 4]``, ``H [T, 16]`` on the same bf16 ``xs [4, T, D]`` / ``out [T, D]``:

    * ``red_ulp`` / ``out_ulp``: max error vs the exact result in bf16 ulps of the largest summand of each element;
    * bias sums ``(num, den)`` of ``sum((dev - rne(ref)) sign(ref)) / sum |ref|`` against the *correctly rounded* exact
      result (HF's fp32 einsum + one bf16 cast): weight-precision defects (the FPU's TF32 truncation of fp32 weights:
      -3.3e-4) show here, while the bias of correct rounding itself does not (e.g. a near-identity H with H_rr just
      below 1 makes X' sit just below the bf16 value X_r, which correctly rounds up: +1.9e-4 vs the unrounded value);
    * ``*_eq``: elements equal to the correctly rounded result."""
    T = xs.shape[1]
    e = dict(red_ulp=0.0, out_ulp=0.0, red_num=0.0, red_den=0.0, out_num=0.0, out_den=0.0, red_eq=0.0, out_eq=0.0,
             red_n=0.0, out_n=0.0)

    def ulp_max(d, big):
        q = d.abs() / _bf16_ulp(big.clamp(min=1e-30))
        return float(q.max()) if bool(torch.isfinite(q).all()) else float("inf")

    for a in range(0, T, chunk):
        sl = slice(a, min(T, a + chunk))
        x = xs[:, sl].double()  # [4, t, D]
        o = out[sl].double()  # [t, D]
        Hm = H[sl].double().reshape(-1, N, N)  # [t, i, j]
        hp = h_pre[sl].double().T[:, :, None]  # [i, t, 1]
        ref_r = torch.zeros_like(x[0])
        big_r = torch.zeros_like(x[0])
        for i in range(N):
            term = hp[i] * x[i]
            ref_r += term
            big_r = torch.maximum(big_r, term.abs())
        ref_o = h_post[sl].double().T[:, :, None] * o[None]  # [i, t, D]: the h_post (x) out term
        big_o = ref_o.abs()
        for j in range(N):
            term = Hm[:, :, j].T[:, :, None] * x[j][None]  # [i, t, D]: H[t, i, j] X_j[t]
            ref_o += term
            big_o = torch.maximum(big_o, term.abs())
        for k, dev, ref, big in (("red", x_red[sl].double(), ref_r, big_r), ("out", x_out[:, sl].double(), ref_o, big_o)):
            rr = _rne_bf16(ref)
            e[f"{k}_ulp"] = max(e[f"{k}_ulp"], ulp_max(dev - ref, big))
            e[f"{k}_num"] += float(((dev - rr) * ref.sign()).sum())
            e[f"{k}_den"] += float(ref.abs().sum())
            e[f"{k}_eq"] += float((dev == rr).double().sum())
            e[f"{k}_n"] += float(ref.numel())
    return e


def maps_errors(p, scalars, h_pre, h_post, H):
    """The coefficient maps alone: device ``h_pre / h_post [T, 4]``, ``H [T, 16]`` against the fp32 golden maps of the
    device's own raw mixes ``p [T, 24]`` (the kernel module's ``motif_mhc_maps_torch`` = ``reference.modules.MHCLayer``
    semantics), and the fraction of pre/post logits beyond +-10 and of res logits beyond +-20 (clamp activity)."""
    km = _km()
    gp, gq, gH = km.motif_mhc_maps_torch(p.float(), scalars, iters=20)
    lp, lq, lr = km.motif_logits_torch(p.float(), scalars)
    return dict(
        maps_H=_maxabs(H, gH),
        maps_h=max(_maxabs(h_pre, gp), _maxabs(h_post, gq)),
        clamp_pp=float((torch.cat([lp, lq], dim=-1).abs() > 10).double().mean()),
        clamp_res=float((lr.abs() > 20).double().mean()),
    )


def _merge_stage(e, mix, maps):
    """Fold per-chip stage checks into ``e`` (max of ulps / maps errors, summed bias sums, mean clamp fractions)."""
    for k in ("red_ulp", "out_ulp"):
        e[f"mix_{k}"] = max(e.get(f"mix_{k}", 0.0), mix[k])
    s = e.setdefault("_bias", [0.0, 0.0, 0.0, 0.0])
    for i, k in enumerate(("red_num", "red_den", "out_num", "out_den")):
        s[i] += mix[k]
    e["mix_red_bias"] = s[0] / max(s[1], 1e-300)
    e["mix_out_bias"] = s[2] / max(s[3], 1e-300)
    q = e.setdefault("_eq", [0.0, 0.0, 0.0, 0.0])
    for i, k in enumerate(("red_eq", "red_n", "out_eq", "out_n")):
        q[i] += mix[k]
    e["mix_red_eq"] = q[0] / max(q[1], 1.0)
    e["mix_out_eq"] = q[2] / max(q[3], 1.0)
    for k in ("maps_H", "maps_h"):
        e[k] = max(e.get(k, 0.0), maps[k])
    n = e.setdefault("_nmaps", [0])
    for k in ("clamp_pp", "clamp_res"):
        e[k] = (e.get(k, 0.0) * n[0] + maps[k]) / (n[0] + 1)
    n[0] += 1


# =====================================================================================================================
# per-site run + comparison
# =====================================================================================================================
def run_site_decode(mesh_device, cfg, site_obj, x32, out32):
    """32 lanes (lane l on DP row l // 8) -> per-chip results [R, C, ...] of p, h_pre, h_post, H, x_red, X'."""
    xh = torch.stack([to_tt_layout(x32[8 * r : 8 * r + 8])[0] for r in range(4)])  # [4, 4, 8, D]
    oh = torch.stack([out32[8 * r : 8 * r + 8][None] for r in range(4)])  # [4, 1, 8, D]
    X = _per_row(mesh_device, cfg, xh)
    O = _per_row(mesh_device, cfg, oh)
    x_red, c = site_obj.pre(X, keep=True)
    Xn = site_obj.post(X, O, c, release=False)
    res = dict(
        p=_chips(c.p, mesh_device)[..., 0, 0, :, :24],
        h_pre=_chips(c.h_pre, mesh_device)[..., :, :4],
        h_post=_chips(c.h_post, mesh_device)[..., :, :4],
        H=_chips(c.H, mesh_device)[..., :, :16],
        x_red=_chips(x_red, mesh_device)[:, :, 0, 0],  # [R, C, 8, D]
        x_out=_chips(Xn, mesh_device)[:, :, 0],  # [R, C, 4, 8, D]
        rep=all(replicas_identical(t, mesh_device, "tp") for t in (c.p, c.h_pre, c.h_post, c.H, x_red, Xn)),
    )
    _free([X, O, x_red, Xn, c])
    return res


def compare_decode(cfg, res, gold32, x32, out32, scalars):
    """End-to-end max errors over all 32 chips (chip (dp, tp) vs lanes 8 dp .. 8 dp + 7), and the mixing / maps stage
    checks on chip (dp, 0) of every DP row (the TP replicas are bitwise identical: ``replicas_identical_tp``)."""
    R, C = cfg.axes.mesh_shape
    e = dict(H=0.0, h_pre=0.0, h_post=0.0, p_rel=0.0, x_red_pcc=1.0, x_out_pcc=1.0, x_red_max=0.0, x_out_max=0.0)
    for r in range(R):
        for c in range(C):
            dp, _ = cfg.axes.roles(r, c)
            sl = slice(8 * dp, 8 * dp + 8)
            gp = gold32["p"][sl]
            e["p_rel"] = max(e["p_rel"], float((res["p"][r, c] - gp).norm() / gp.norm()))
            e["H"] = max(e["H"], _maxabs(res["H"][r, c], gold32["H"][sl]))
            e["h_pre"] = max(e["h_pre"], _maxabs(res["h_pre"][r, c], gold32["h_pre"][sl]))
            e["h_post"] = max(e["h_post"], _maxabs(res["h_post"][r, c], gold32["h_post"][sl]))
            gx = gold32["x_red"][sl]
            e["x_red_pcc"] = min(e["x_red_pcc"], _pcc(res["x_red"][r, c], gx))
            e["x_red_max"] = max(e["x_red_max"], _maxabs(res["x_red"][r, c], gx))
            go = gold32["x_out"][sl].permute(1, 0, 2)  # [4, 8, D]
            e["x_out_pcc"] = min(e["x_out_pcc"], _pcc(res["x_out"][r, c], go))
            e["x_out_max"] = max(e["x_out_max"], _maxabs(res["x_out"][r, c], go))
    for dp in range(cfg.axes.dp_size):
        r, c = cfg.axes.coord(dp, 0)
        sl = slice(8 * dp, 8 * dp + 8)
        mix = mix_errors(x32[sl].permute(1, 0, 2), out32[sl], res["h_pre"][r, c], res["h_post"][r, c], res["H"][r, c],
                         res["x_red"][r, c], res["x_out"][r, c])
        maps = maps_errors(res["p"][r, c], scalars, res["h_pre"][r, c], res["h_post"][r, c], res["H"][r, c])
        _merge_stage(e, mix, maps)
    e["replicas_identical_tp"] = res["rep"]
    return e


def run_site_prefill(mesh_device, site_obj, x, out, S):
    """Replicated prefill of ``x [T, 4, D]`` padded with zero tokens to the bucket ``S`` (``S=None``: logical T as is,
    tile-padded by TTNN); results of chip 0 [:T] and the all-32-chip replica check."""
    T = x.shape[0]
    S = T if S is None else S
    xp = torch.zeros(S, N, D, dtype=torch.bfloat16)
    xp[:T] = x
    op = torch.zeros(S, D, dtype=torch.bfloat16)
    op[:T] = out
    X = _replicated(mesh_device, to_tt_layout(xp))
    O = _replicated(mesh_device, op[None, None])
    t0 = time.perf_counter()
    x_red, c = site_obj.pre(X, keep=True)
    Xn = site_obj.post(X, O, c, release=False)
    ttnn.synchronize_device(mesh_device)
    eager_ms = (time.perf_counter() - t0) * 1e3
    d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()  # noqa: E731
    res = dict(
        p=d0(c.p)[0, 0, :T, :24],
        h_pre=d0(c.h_pre)[:T, :4],
        h_post=d0(c.h_post)[:T, :4],
        H=d0(c.H)[:T, :16],
        x_red=d0(x_red)[0, 0, :T],
        x_out=d0(Xn)[0, :, :T],
        eager_ms=eager_ms,
        rep=all(_replicas_all(t, mesh_device) for t in (c.p, c.h_pre, c.h_post, c.H, x_red, Xn)),
    )
    _free([X, O, x_red, Xn, c])
    return res


def compare_prefill(res, gold, x, out, scalars):
    gp = gold["p"]
    e = dict(
        p_rel=float((res["p"] - gp).norm() / gp.norm()),
        H=_maxabs(res["H"], gold["H"]),
        h_pre=_maxabs(res["h_pre"], gold["h_pre"]),
        h_post=_maxabs(res["h_post"], gold["h_post"]),
        x_red_pcc=_pcc(res["x_red"], gold["x_red"]),
        x_red_max=_maxabs(res["x_red"], gold["x_red"]),
        x_out_pcc=_pcc(res["x_out"], gold["x_out"].permute(1, 0, 2)),
        x_out_max=_maxabs(res["x_out"], gold["x_out"].permute(1, 0, 2)),
    )
    mix = mix_errors(x.permute(1, 0, 2), out, res["h_pre"], res["h_post"], res["H"], res["x_red"], res["x_out"])
    _merge_stage(e, mix, maps_errors(res["p"], scalars, res["h_pre"], res["h_post"], res["H"]))
    e["replicas_identical_all"] = res["rep"]
    return e


def _fmt(e):
    def f(k, v):
        if isinstance(v, bool) or not isinstance(v, float):
            return f"{k}={v}"
        return f"{k}={v:.7f}" if k.endswith("_pcc") else f"{k}={v:.3e}"

    return " ".join(f(k, v) for k, v in e.items() if not k.startswith("_"))


def _ok(e, backend, regime="real", *, truncated_weights=False, kind="real"):
    """All per-site checks (see the module docstring). ``regime="peaked"``: end-to-end H / h and PCC are reported only
    (x20-x40 alphas amplify the FPU floor of p into ~1e-2 on H); the stage checks always apply.

    Mixing ulp bound: a correctly rounded C-term sum is off by <= 0.5 ulp(result) <= C - 1 ulps of the largest summand
    (|result| <= C max), nearest-even TF32 weights add <= C/8, so C holds (worst case 4.625 for C = 5).
    ``truncated_weights`` (``glue="ttnn"`` with ``mix="wr"``: the FPU truncates the unrounded fp32 weights) adds <= C/4
    instead and is biased by construction: ulp bound C + C/4, bias reported only."""
    rep = e.get("replicas_identical_tp", e.get("replicas_identical_all"))
    good = bool(rep)
    slack = 1.25 if truncated_weights else 1.0
    good &= e["mix_red_ulp"] <= TH_MIX_ULP["red"] * slack and e["mix_out_ulp"] <= TH_MIX_ULP["out"] * slack
    if not truncated_weights:
        good &= abs(e["mix_red_bias"]) <= TH_MIX_BIAS and abs(e["mix_out_bias"]) <= TH_MIX_BIAS
    if backend == "motif":
        good &= e["maps_H"] <= TH_MAPS_MOTIF[0] and e["maps_h"] <= TH_MAPS_MOTIF[1]
    elif regime != "peaked":
        good &= e["maps_H"] <= TH_MAPS_STOCK[0] and e["maps_h"] <= TH_MAPS_STOCK[1]
    if regime == "peaked":
        good &= e["clamp_pp"] > 0 and e["clamp_res"] > 0  # the regime must exercise both clamps
        return bool(good)
    good &= e["p_rel"] <= TH_P_REL[kind]
    good &= e["x_red_pcc"] >= TH_PCC and e["x_out_pcc"] >= TH_PCC
    if backend == "motif":
        good &= e["H"] <= TH_H and e["h_pre"] <= TH_PRE and e["h_post"] <= TH_POST
    else:
        good &= e["H"] <= TH_H_STOCK and e["h_pre"] <= TH_H_STOCK_PP and e["h_post"] <= TH_H_STOCK_PP
    return bool(good)


# =====================================================================================================================
# tests
# =====================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_kernels(mesh_device):
    """The generic_op kernels of tt/mhc.py in isolation (fast, run first): ``finalize_mixes`` (decode split-K and
    multi-core prefill layouts) vs fp64, ``coefficient_layout`` (with / without the TF32 rounding) vs torch bit exact,
    ``post_mix`` vs the stock weighted reduce on the concat (bitwise); plus the zero-copy views, the strict Sinkhorn
    backend selection and the mesh-keyed shared constants."""
    from models.demos.motif3.tt import mhc as M

    log_fabric(mesh_device, "test_mhc_kernels")
    cfg = _cfg(mesh_device)
    g = torch.Generator().manual_seed(9)
    d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()  # noqa: E731
    failures = []
    eps = 16384 * 1e-6

    # ---- Sinkhorn backend: the kernel module must import; the default site uses it (no silent fallback) ----
    km = M.motif_kernel_module()
    assert km is not None, f"tt/kernels/sinkhorn_motif failed to import: {M._KM_IMPORT_ERROR!r}"
    src = random_site_source(2, "mhc_attn", seed=3)
    st = M.MHCSite(mesh_device, cfg, 2, "mhc_attn", source=src, cache=False)
    print(f"[mhc] default backend: {st.sinkhorn} (mix={st.mix}, glue={st.glue}, finalize={st.finalize}, "
          f"post_concat={st.post_concat})", flush=True)
    assert st.sinkhorn == "motif"
    for bad in ("auto", "direct"):
        with pytest.raises(ValueError):
            M.MHCSite(mesh_device, cfg, 2, "mhc_attn", source=src, cache=False, sinkhorn=bad)

    # ---- zero-copy views (the module relies on them; README §10 rule 2) ----
    x32, o32 = random_streams(32, seed=21)
    xh = torch.stack([to_tt_layout(x32[8 * r : 8 * r + 8])[0] for r in range(4)])
    X = _per_row(mesh_device, cfg, xh)
    X32 = M._view_logical(X, [1, N, 32, D])
    xv = ttnn.experimental.view(X32, [1, st.decode_split, 32, N * D // st.decode_split])
    views = dict(x_rows=X32.buffer_address() == X.buffer_address(), x_split=xv.buffer_address() == X.buffer_address(),
                 proj_split=st.proj_split.buffer_address() == st.proj.buffer_address())
    print(f"[mhc] zero-copy views: {views}", flush=True)
    if not all(views.values()):
        failures.append(f"views not zero-copy: {views}")

    # ---- shared constants keyed by the process-unique mesh id ----
    ss = M.MHCSite(mesh_device, cfg, 2, "mhc_attn", source=src, cache=False, sinkhorn="stock")
    keys = [k for k in M._SHARED if k[1] == "preclamped_consts"]
    want_uid = ("mesh", int(mesh_device.id()))
    print(f"[mhc] _SHARED keys: {sorted(M._SHARED)}; this mesh {want_uid}", flush=True)
    if (want_uid, "preclamped_consts") not in M._SHARED or any(k[0][0] != "mesh" for k in keys):
        failures.append(f"_SHARED not keyed by MeshDevice.id(): {sorted(M._SHARED)}")
    ss.release()
    M.release_shared(mesh_device)
    if any(k[0] == want_uid for k in M._SHARED):
        failures.append("release_shared left entries of this mesh")
    ss = M.MHCSite(mesh_device, cfg, 2, "mhc_attn", source=src, cache=False, sinkhorn="stock")  # recreated on demand
    if not (ss.stock_consts.is_allocated() and (want_uid, "preclamped_consts") in M._SHARED):
        failures.append("shared consts not recreated after release_shared")
    ss.release()

    # ---- finalize_mixes: decode split-K layout (1 core) and the prefill layout (diagonal tiles, multi-core) ----
    y = torch.randn(1, 32, 32, 32, generator=g)
    sq = torch.rand(1, 32, 32, 32, generator=g) * 50
    yt, st_ = _replicated(mesh_device, y, ttnn.float32), _replicated(mesh_device, sq, ttnn.float32)
    p = M.finalize_mixes(yt, st_, eps=eps, T=8, memory_config=ttnn.L1_MEMORY_CONFIG)
    want = y.double().sum(1)[0] * torch.rsqrt(sq.double().sum(1)[0][:, :1] + eps)
    got = d0(p)[0, 0]
    err = float((got[:8] - want[:8]).abs().max() / want[:8].abs().max())
    print(f"[mhc] finalize_mixes decode: shape {list(p.shape)} max rel err {err:.3e} (fp32 rounding ~1e-7)", flush=True)
    if not (list(p.shape) == [1, 1, 8, 32] and err < 1e-6):
        failures.append(f"finalize_mixes decode err {err:.3e}")
    _free([yt, st_, p])
    for S, T in ((256, 256), (8192, 8192), (224, 200)):  # 8 / 256 token tiles (> 120 cores), logical T = 200
        St = S // 32
        yw = torch.randn(1, 1, 4 * S, 128, generator=g)  # projection output: diagonal blocks are the stream partials
        sw = torch.rand(1, 4, S, 32, generator=g) * 50
        diag = torch.stack([yw[0, 0, s * S : (s + 1) * S, 32 * s : 32 * s + 32] for s in range(4)])  # [4, S, 32]
        want = diag.double().sum(0) * torch.rsqrt(sw.double()[0, :, :, :1].sum(0) + eps)  # [S, 32]
        yt = _replicated(mesh_device, yw, ttnn.float32)
        st_ = _replicated(mesh_device, sw, ttnn.float32)
        p = M.finalize_mixes(yt, st_, eps=eps, T=T, nb=4, n_tiles=St, y_strides=(4 * St + 1, 4), s_strides=(St, 1),
                             chunk=4)
        got = d0(p)[0, 0]
        err = float((got[:T] - want[:T]).abs().max() / want[:T].abs().max())
        rep = _replicas_all(p, mesh_device)
        print(f"[mhc] finalize_mixes prefill S={S} T={T} ({St} tiles): shape {list(p.shape)} max rel err {err:.3e}, "
              f"replicas identical {rep}", flush=True)
        if not (list(p.shape) == [1, 1, T, 32] and err < 1e-6 and rep):
            failures.append(f"finalize_mixes prefill S={S} err {err:.3e} rep {rep}")
        _free([yt, st_, p])

    # ---- coefficient_layout: bit exact (column 0), with and without the TF32 rounding ----
    hp = torch.rand(32, 4, generator=g)
    hq = torch.rand(32, 4, generator=g) * 2
    H = torch.rand(32, 16, generator=g)
    tt = [_replicated(mesh_device, t.reshape(32, -1), ttnn.float32) for t in (hp, hq, H)]
    tt = [ttnn.reshape(t, ttnn.Shape([8, int(t.shape[-1])]), ttnn.Shape([32, 32])) for t in tt]
    for halve in (False, True):
        for rnd in (False, True):
            w_pre, w_post = M.coefficient_layout(*tt, post_halve=halve, round_tf32=rnd,
                                                 memory_config=ttnn.L1_MEMORY_CONFIG)
            q = M.tf32_rne if rnd else (lambda t: t)
            wp, wq = d0(w_pre)[..., 0], d0(w_post)[..., 0]  # [1,4,8], [4,5,8]
            ok = torch.equal(wp[0], q(hp[:8]).double().T) and torch.equal(
                wq[:, :4], q(H[:8]).double().T.reshape(4, 4, 8)) and torch.equal(
                wq[:, 4], q(hq[:8] * (0.5 if halve else 1.0)).double().T)
            print(f"[mhc] coefficient_layout post_halve={halve} round_tf32={rnd}: shapes {list(w_pre.shape)} "
                  f"{list(w_post.shape)} exact={ok}", flush=True)
            if not ok:
                failures.append(f"coefficient_layout post_halve={halve} round_tf32={rnd} not bit exact")
            _free([w_pre, w_post])
    # host model of the rounding: ties to even, carries, Inf / NaN
    one = 1.0
    cases = torch.tensor([one + 2.0**-11, one + 3 * 2.0**-11, one + 2.0**-11 + 2.0**-20, 2.0 - 2.0**-23,
                          float("inf"), -(one + 3 * 2.0**-11)], dtype=torch.float32)
    want_r = torch.tensor([one, one + 2.0**-9, one + 2.0**-10, 2.0, float("inf"), -(one + 2.0**-9)])
    if not torch.equal(M.tf32_rne(cases), want_r) or not bool(torch.isnan(M.tf32_rne(torch.tensor([float("nan")])))):
        failures.append(f"tf32_rne host model wrong: {M.tf32_rne(cases).tolist()}")

    # ---- post_mix == attn_res_weighted_reduce_nc(concat([X, out])) bitwise; decode (L1 weights) and prefill ----
    WR = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc
    for T, S, wmc in ((8, 32, ttnn.L1_MEMORY_CONFIG), (256, 256, ttnn.DRAM_MEMORY_CONFIG),
                      (200, 224, ttnn.DRAM_MEMORY_CONFIG)):
        xs = (torch.randn(1, 4, S, D, generator=g) * 2).bfloat16()
        os_ = (torch.randn(1, 1, S, D, generator=g) * 2).bfloat16()
        w = M.tf32_rne(torch.rand(4, 5, S, 1, generator=g))
        Xt = _replicated(mesh_device, xs)
        Ot = _replicated(mesh_device, os_)
        Wt = _replicated(mesh_device, w, ttnn.float32, memory_config=wmc)
        if T != S:
            Xt, Ot, Wt = (M._view_logical(t, [int(t.shape[0]), int(t.shape[1]), T, int(t.shape[3])])
                          for t in (Xt, Ot, Wt))
        a = M.post_mix(Xt, Ot, Wt, compute_role=cfg.compute_role("mhc"))
        xc = ttnn.concat([Xt, Ot], dim=1)
        b = ttnn.reshape(WR(xc, Wt, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            compute_kernel_config=cfg.compute_config("mhc")), [1, 4, T, D])
        ga, gb = d0(a), d0(b)
        ref = torch.einsum("ctd,rct->rtd", torch.cat([xs, os_], 1)[0, :, :T].double(), w[:, :, :T, 0].double())
        same = torch.equal(ga, gb)
        rel = float((ga[0] - ref).norm() / ref.norm())
        print(f"[mhc] post_mix T={T} (padded {S}) weights {'L1' if wmc == ttnn.L1_MEMORY_CONFIG else 'DRAM'}: shape "
              f"{list(a.shape)}, bitwise == stock op on the concat: {same}, rel err vs fp64 {rel:.3e}", flush=True)
        if not (same and list(a.shape) == [1, 4, T, D] and rel < 3e-3):
            failures.append(f"post_mix T={T}: bitwise {same} rel {rel:.3e}")
        _free([Xt, Ot, Wt, a, xc, b])
    _free(X)
    st.release()
    assert not failures, "\n".join(failures)


class _NoSource:
    """A weight source that must not be touched (cache hits only)."""

    def has(self, name):
        raise AssertionError(f"weight source touched on a cache hit: has({name})")

    def get(self, name, dtype=None):
        raise AssertionError(f"weight source touched on a cache hit: get({name})")


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_cache_roundtrip(mesh_device):
    """TT weight cache: a site built with cache=True writes its tensors; a second site loads them without touching the
    source and gives bitwise-identical decode outputs (both backends). The cache names are the module's own (the
    x128-scaled projection is not stored under the README's "<site>.blocks" name). Files under tt_cache/test/mhc/cache
    (removed)."""
    import shutil

    from models.demos.motif3.tt.mhc import PROJ_CACHE_NAME, MHCSite

    log_fabric(mesh_device, "test_mhc_cache_roundtrip")
    root = DATA / "cache"
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device, tt_cache_root=root)
    src = random_site_source(3, "mhc_ffn", seed=17)
    x32, o32 = random_streams(32, seed=23)
    want = {
        "motif": {PROJ_CACHE_NAME, "motif_consts"},
        "stock": {PROJ_CACHE_NAME, "motif_consts", "alpha_row", "bias_row", "lo_row", "hi_row"},
    }
    try:
        for backend in BACKENDS:
            a = MHCSite(mesh_device, cfg, 3, "mhc_ffn", source=src, cache=True, sinkhorn=backend)
            ra = run_site_decode(mesh_device, cfg, a, x32, o32)
            files = sorted(str(f.relative_to(root)) for f in root.rglob("*.tensorbin"))
            names = {Path(f).name.split("__")[0].split(".", 1)[1] for f in files}
            b = MHCSite(mesh_device, cfg, 3, "mhc_ffn", source=_NoSource(), cache=True, sinkhorn=backend)
            rb = run_site_decode(mesh_device, cfg, b, x32, o32)
            same = all(torch.equal(ra[k], rb[k]) for k in ("p", "h_pre", "h_post", "H", "x_red", "x_out"))
            print(f"[mhc] cache {backend}: {len(files)} files {sorted(names)}, reload bitwise identical: {same}",
                  flush=True)
            assert same and names == want[backend], (names, want[backend])
            a.release()
            b.release()
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_random(mesh_device):
    """Random weights at real dims: decode (per-row lanes) and prefill, realistic and peaked logits, both backends."""
    from models.demos.motif3.tt.mhc import MHCSite

    log_fabric(mesh_device, "test_mhc_random")
    cfg = _cfg(mesh_device)
    failures, lines = [], []
    for ri, regime in enumerate(("realistic", "peaked")):
        for si, site in enumerate(("mhc_attn", "mhc_ffn")):
            layer = 2
            src = random_site_source(layer, site, seed=10 * ri + si, regime=regime)
            proj, gamma, sc = site_tensors(src, W.hf_name(layer, site))
            m64 = ref_layer(proj, gamma, sc)
            x32, o32 = random_streams(32, seed=7)
            gold = golden(m64, x32, o32)
            xs, os_ = random_streams(200, seed=11)
            g2 = golden(m64, xs, os_)
            for backend in BACKENDS:
                extra = regime == "realistic" and site == "mhc_attn"
                variants = ((("wr", "kernel", False), ("wr", "kernel", True), ("composite", "kernel", False),
                             ("wr", "ttnn", False)) if extra else (("wr", "kernel", False),))
                for mix, glue, pc in variants:
                    st = MHCSite(mesh_device, cfg, layer, site, source=src, cache=False, sinkhorn=backend, mix=mix,
                                 glue=glue, post_concat=pc)
                    e = compare_decode(cfg, run_site_decode(mesh_device, cfg, st, x32, o32), gold, x32, o32, sc)
                    # glue="ttnn" feeds unrounded fp32 weights to the FPU (TF32 truncation): bias reported, not bounded
                    trunc = mix == "wr" and glue == "ttnn"
                    tag = f"random/{regime}/{site}/{backend}/{mix}/glue={glue}/post_concat={pc}/decode"
                    ok = _ok(e, backend, regime, truncated_weights=trunc, kind="random")
                    lines.append(f"{'PASS' if ok else 'FAIL'} {tag}: {_fmt(e)}")
                    if not ok:
                        failures.append(lines[-1])
                    if mix == "wr" and glue == "kernel" and not pc:
                        for S_ in (256, None):  # bucket-padded, and logical T = 200 (tile-padded to 224 by TTNN)
                            e2 = compare_prefill(run_site_prefill(mesh_device, st, xs, os_, S_), g2, xs, os_, sc)
                            tag2 = f"random/{regime}/{site}/{backend}/prefill S={S_ or 'T=200 logical'} (200 real)"
                            ok2 = _ok(e2, backend, regime, kind="random")
                            lines.append(f"{'PASS' if ok2 else 'FAIL'} {tag2}: {_fmt(e2)}")
                            if not ok2:
                                failures.append(lines[-1])
                    st.release()
    for line in lines:
        print("[mhc] " + line, flush=True)
    assert not failures, "\n".join(failures)


def _real_decode_data():
    path = DATA / "real_decode_tokens.pt"
    if not path.is_file():
        pytest.skip(f"{path} missing: run `python -m models.demos.motif3.tests.unit.test_mhc --capture` (host)")
    return torch.load(path, weights_only=True)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_real_decode(mesh_device):
    """MHC-5: real weights and streams, layers 0-5 and 28-35, both sites, decode (32 lanes over the 4 DP rows)."""
    from models.demos.motif3.tt.mhc import MHCSite

    log_fabric(mesh_device, "test_mhc_real_decode")
    cfg = _cfg(mesh_device)
    data = _real_decode_data()
    src = W.HFWeightLoader()
    summary, failures = {}, []
    order = lambda kv: (kv[0].split("|")[0], int(kv[0].split("|")[1]), kv[0])  # noqa: E731
    for key, d in sorted(data["sites"].items(), key=order):
        prompt, layer, site = key.split("|")
        layer = int(layer)
        if layer not in REAL_DECODE_LAYERS or not src.layer_available(layer):
            continue
        proj, gamma, sc = site_tensors(src, W.hf_name(layer, site))
        gold = golden(ref_layer(proj, gamma, sc), d["x"], d["out"])
        gold_bf16 = golden(ref_layer(proj, gamma, sc, torch.bfloat16), d["x"], d["out"], torch.bfloat16)
        hf_vs_fp64 = _maxabs(gold_bf16["H"], gold["H"])
        for backend in BACKENDS:
            st = MHCSite(mesh_device, cfg, layer, site, source=src, cache=False, sinkhorn=backend)
            e = compare_decode(cfg, run_site_decode(mesh_device, cfg, st, d["x"], d["out"]), gold, d["x"], d["out"], sc)
            st.release()
            e["hf_bf16_vs_fp64_H"] = hf_vs_fp64
            ok = _ok(e, backend)
            line = f"{'PASS' if ok else 'FAIL'} real/{prompt}/L{layer:02d}/{site}/{backend}/decode: {_fmt(e)}"
            print("[mhc] " + line, flush=True)
            s = summary.setdefault(backend, dict(H=0.0, h_pre=0.0, h_post=0.0, p_rel=0.0, x_red_pcc=1.0, x_out_pcc=1.0,
                                                 mix_red_ulp=0.0, mix_out_ulp=0.0, maps_H=0.0, maps_h=0.0, worst_H="",
                                                 sites=0, _bias=[0.0, 0.0, 0.0, 0.0]))
            s["sites"] += 1
            if e["H"] > s["H"]:
                s["worst_H"] = f"{prompt}/L{layer}/{site}"
            for k in ("H", "h_pre", "h_post", "p_rel", "mix_red_ulp", "mix_out_ulp", "maps_H", "maps_h"):
                s[k] = max(s[k], e[k])
            for k in ("x_red_pcc", "x_out_pcc"):
                s[k] = min(s[k], e[k])
            for i in range(4):
                s["_bias"][i] += e["_bias"][i]
            if not ok:
                failures.append(line)
    for b, s in summary.items():
        bias = s.pop("_bias")
        s["mix_red_bias_all_sites"] = bias[0] / bias[1]
        s["mix_out_bias_all_sites"] = bias[2] / bias[3]
        print(f"[mhc] SUMMARY real decode {b}: {json.dumps(s)}", flush=True)
        if max(abs(s["mix_red_bias_all_sites"]), abs(s["mix_out_bias_all_sites"])) > TH_MIX_BIAS_AGG:
            failures.append(f"{b}: aggregate mixing bias over all sites {s['mix_red_bias_all_sites']:.3e} / "
                            f"{s['mix_out_bias_all_sites']:.3e} > {TH_MIX_BIAS_AGG:.1e}")
    assert summary, "no real layer available"
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_real_prefill(mesh_device):
    """Real weights/streams at prefill: default prompt L0/2/4 (both sites, S=256), long prompt L30 mhc_attn
    (S = 2048 with 2027 real tokens, S = 4096 with the tokens tiled)."""
    from models.demos.motif3.tt.mhc import MHCSite

    log_fabric(mesh_device, "test_mhc_real_prefill")
    cfg = _cfg(mesh_device)
    src = W.HFWeightLoader()
    cases = []
    for layer in (0, 2, 4):
        f = DATA / f"real_prefill_default_L{layer:02d}.pt"
        if f.is_file() and src.layer_available(layer):
            d = torch.load(f, weights_only=True)
            cases.append((f"default/L{layer:02d}/mhc_attn", layer, "mhc_attn", d["x_in"], d["attn_out"], 256))
            cases.append((f"default/L{layer:02d}/mhc_ffn", layer, "mhc_ffn", d["x_mid"], d["ffn_out"], 256))
    f = DATA / "real_prefill_card_L30_mhc_attn.pt"
    if f.is_file() and src.layer_available(30):
        d = torch.load(f, weights_only=True)
        cases.append(("card/L30/mhc_attn S=2048", 30, "mhc_attn", d["x"], d["out"], 2048))
        xt = torch.cat([d["x"], d["x"]])[:4096]
        ot = torch.cat([d["out"], d["out"]])[:4096]
        cases.append(("card/L30/mhc_attn S=4096 (tiled)", 30, "mhc_attn", xt, ot, 4096))
    if not cases:
        pytest.skip("no real prefill data / layers")
    failures = []
    for name, layer, site, x, out, S in cases:
        proj, gamma, sc = site_tensors(src, W.hf_name(layer, site))
        gold = golden(ref_layer(proj, gamma, sc), x, out)
        for backend in BACKENDS:
            st = MHCSite(mesh_device, cfg, layer, site, source=src, cache=False, sinkhorn=backend)
            res = run_site_prefill(mesh_device, st, x, out, S)
            e = compare_prefill(res, gold, x, out, sc)
            e["eager_ms"] = res["eager_ms"]
            st.release()
            ok = _ok(e, backend)
            line = f"{'PASS' if ok else 'FAIL'} real/{name}/{backend}/prefill T={x.shape[0]}: {_fmt(e)}"
            print("[mhc] " + line, flush=True)
            if not ok:
                failures.append(line)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_decode_trace_latency(mesh_device):
    """Decode trace capture + replay with new per-row inputs (equal to eager), the layout kernel against the ttnn glue,
    and the per-site latency breakdown."""
    from models.demos.motif3.tt import mhc as M

    log_fabric(mesh_device, "test_mhc_decode_trace_latency")
    cfg = _cfg(mesh_device)
    src = random_site_source(2, "mhc_attn", seed=3)
    failures = []
    variants = [  # (backend, glue, mix_l1, l1_intermediates, finalize, decode_split, post_concat); first = the defaults
        ("motif", "kernel", True, True, "kernel", 32, False),
        ("motif", "kernel", True, True, "kernel", 32, True),
        ("motif", "kernel", True, True, "kernel", 16, False),
        ("motif", "kernel", True, True, "ttnn", 32, False),
        ("motif", "kernel", False, False, "ttnn", 32, False),
        ("motif", "ttnn", False, False, "ttnn", 32, False),
        ("stock", "kernel", True, True, "kernel", 32, False),
    ]
    for backend, glue, mix_l1, l1i, fin, nbk, pc in variants:
        st = M.MHCSite(mesh_device, cfg, 2, "mhc_attn", source=src, cache=False, sinkhorn=backend, glue=glue,
                       mix_l1=mix_l1, l1_intermediates=l1i, finalize=fin, decode_split=nbk, post_concat=pc)
        tag = f"{backend}/glue={glue}/mix_l1={mix_l1}/l1_int={l1i}/finalize={fin}/split={nbk}/post_concat={pc}"
        x32, o32 = random_streams(32, seed=21)
        xh = torch.stack([to_tt_layout(x32[8 * r : 8 * r + 8])[0] for r in range(4)])
        oh = torch.stack([o32[8 * r : 8 * r + 8][None] for r in range(4)])
        X = _per_row(mesh_device, cfg, xh)
        O = _per_row(mesh_device, cfg, oh)

        def step():
            x_red, c = st.pre(X)
            return [x_red, st.post(X, O, c)]

        # ---- trace safety: capture once, replay with new inputs, compare with eager ----
        _free(step())  # eager warm-up: the generic_op binaries must reach the device before the capture
        ttnn.synchronize_device(mesh_device)
        with _Capture(mesh_device) as cap:
            touts = step()
        try:
            for it in range(2):
                x2, o2 = random_streams(32, seed=100 + it)
                xh2 = torch.stack([to_tt_layout(x2[8 * r : 8 * r + 8])[0] for r in range(4)])
                oh2 = torch.stack([o2[8 * r : 8 * r + 8][None] for r in range(4)])
                mapper = W.mesh_mapper(mesh_device, cfg.axes, dp_dim=0)
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(xh2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper), X
                )
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(oh2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper), O
                )
                ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                eager = step()
                same = all(torch.equal(_chips(a, mesh_device), _chips(b, mesh_device)) for a, b in zip(touts, eager))
                print(f"[mhc] trace replay {tag} it{it}: traced == eager bitwise: {same}", flush=True)
                if not same:
                    failures.append(f"{tag}: trace replay differs from eager (it {it})")
                _free(eager)
        finally:
            ttnn.release_trace(mesh_device, cap.tid)

        # ---- layout kernel == ttnn glue (column 0 of every weight tile; the kernel rounds to TF32 for mix="wr") ----
        if glue == "kernel":
            ref_site = M.MHCSite(mesh_device, cfg, 2, "mhc_attn", source=src, cache=False, sinkhorn=backend, glue="ttnn",
                                 mix_l1=False, l1_intermediates=False, finalize=fin, decode_split=nbk)
            ck, ct = st.coefficients(X), ref_site.coefficients(X)
            wa = [_chips(t, mesh_device)[..., 0] for t in (ck.w_pre, ck.w_post)]
            wb = [M.tf32_rne(_chips(t, mesh_device)[..., 0].float()).double() for t in (ct.w_pre, ct.w_post)]
            same_w = all(torch.equal(a, b) for a, b in zip(wa, wb))
            print(f"[mhc] coefficients {tag} == tf32_rne(ttnn glue): bitwise={same_w}", flush=True)
            if not same_w:
                failures.append(f"{tag}: layout kernel differs from the (rounded) ttnn glue")
            _free([ck, ct])
            ref_site.release()

        # ---- latency breakdown (decode, one site) ----
        X32 = M._view_logical(X, [1, 4, 32, D])
        xv = ttnn.experimental.view(X32, [1, st.decode_split, 32, 4 * D // st.decode_split])
        lat = {}
        lat["proj_matmul"] = _traced_us(
            mesh_device,
            lambda: ttnn.matmul(xv, st.proj_split, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=st.ckc,
                                program_config=st.pc_decode),
        )
        lat["ss_stats"] = _traced_us(
            mesh_device, lambda: ttnn.rms_norm_pre_all_gather(xv, dtype=ttnn.float32, compute_kernel_config=st.ckc)
        )
        # accuracy of p for this variant (vs fp64 of the same random site), on the original inputs
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=W.mesh_mapper(mesh_device, cfg.axes, dp_dim=0)), X
        )
        proj, gamma, sc = site_tensors(src, W.hf_name(2, "mhc_attn"))
        gold_p = golden(ref_layer(proj, gamma, sc), x32, o32)["p"]
        pv = _chips(st.mixes(X), mesh_device)[:, :, 0, 0, :, :24]
        prel = 0.0
        for r in range(4):
            for c in range(8):
                gp = gold_p[8 * cfg.axes.roles(r, c)[0] : 8 * cfg.axes.roles(r, c)[0] + 8]
                prel = max(prel, float((pv[r, c] - gp).norm() / gp.norm()))
        print(f"[mhc] p_rel {tag}: {prel:.3e}", flush=True)
        lat["mixes_total"] = _traced_us(mesh_device, lambda: st.mixes(X))
        p = st.mixes(X)
        lat["maps"] = _traced_us(mesh_device, lambda: list(st.maps(p)))
        lat["coefficients_total"] = _traced_us(mesh_device, lambda: st.coefficients(X))
        c = st.coefficients(X)
        lat["apply_pre"] = _traced_us(mesh_device, lambda: st.apply_pre(X, c))
        lat["apply_post"] = _traced_us(mesh_device, lambda: st.apply_post(X, O, c))
        lat["pre_total"] = _traced_us(mesh_device, lambda: list(st.pre(X)))
        lat["site_total"] = _traced_us(mesh_device, step, n=16)
        lat["site_eager"] = _eager_us(mesh_device, step)
        lat["glue"] = lat["coefficients_total"] - lat["mixes_total"] - lat["maps"]
        print(f"[mhc] LATENCY {tag} {json.dumps({k: round(v, 1) for k, v in lat.items()})}", flush=True)
        _free([p, c, X, O, touts])
        st.release()
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_decode_fused(mesh_device):
    """Phase C D3 (``MOTIF3_MHC_DECODE=fused``, ``tt/kernels/mhc_decode.py``): ``MHCSite(decode="fused")`` against
    ``decode="ops"`` through ``pre`` / ``post`` -- x_red and X' bitwise on all 32 chips, logical AND padded rows -- at
    T = 8 (T32), 16 (T64 rows) and 32 (the full-tile finalize), random realistic / peaked sites, a trace capture
    replayed with new inputs; the packed tile against the release weight tiles (all 32 token rows); real weights and
    streams on the real decode sites (when the captured data and the layers are available); traced site latency."""
    from models.demos.motif3.tt import mhc as M
    from models.demos.motif3.tt.kernels import mhc_decode as MD

    log_fabric(mesh_device, "test_mhc_decode_fused")
    cfg = _cfg(mesh_device)
    mapper = W.mesh_mapper(mesh_device, cfg.axes, dp_dim=0)
    failures = []

    def padded(t, shape):
        return _chips(M._view_logical(t, shape), mesh_device)

    def upload(x, o, L):
        xh = torch.stack([to_tt_layout(x[L * r : L * r + L])[0] for r in range(4)])
        oh = torch.stack([o[L * r : L * r + L][None] for r in range(4)])
        kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        return xh, oh, kw

    def compare(tag, ops, fus, X, O):
        a_red, ca = ops.pre(X)
        a_out = ops.post(X, O, ca)
        b_red, cb = fus.pre(X)
        assert cb.packed is not None and cb.w_pre is None, "fused site did not take the fused path"
        b_out = fus.post(X, O, cb)
        same = {
            "x_red": torch.equal(_chips(a_red, mesh_device), _chips(b_red, mesh_device)),
            "x_out": torch.equal(_chips(a_out, mesh_device), _chips(b_out, mesh_device)),
            "x_red_padded": torch.equal(padded(a_red, [1, 1, 32, D]), padded(b_red, [1, 1, 32, D])),
            "x_out_padded": torch.equal(padded(a_out, [1, N, 32, D]), padded(b_out, [1, N, 32, D])),
        }
        print(f"[mhc] decode fused {tag}: {same}", flush=True)
        if not all(same.values()):
            failures.append(f"{tag}: {same}")
        _free([a_red, a_out, b_red, b_out])

    for regime, seed in (("realistic", 3), ("peaked", 5)):
        src = random_site_source(2, "mhc_ffn", seed=seed, regime=regime)
        ops = M.MHCSite(mesh_device, cfg, 2, "mhc_ffn", source=src, cache=False, decode="ops")
        fus = M.MHCSite(mesh_device, cfg, 2, "mhc_ffn", source=src, cache=False, decode="fused")
        assert fus.decode_fused and not ops.decode_fused
        for L in (8, 16, 32):
            x, o = random_streams(4 * L, seed=40 + seed + L)
            xh, oh, kw = upload(x, o, L)
            X = ttnn.from_torch(xh, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
            O = ttnn.from_torch(oh, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
            tag = f"{regime}/T={L}"
            compare(tag, ops, fus, X, O)
            # the packed tile == the release weights, every token row (incl. the padding rows)
            c = ops.coefficients(X)
            wpre = padded(c.w_pre, [1, N, 32, 1])[..., 0]
            wpost = padded(c.w_post, [N, N + 1, 32, 1])[..., 0]
            y, s = fus._partials_decode(X)
            P = MD.coefficients_packed(y, s, fus.motif_consts, eps=fus.ss_eps, T=L)
            Ph = _chips(P, mesh_device).reshape(4, 8, MD.DEFAULT_NCOPY, 32, 32)
            exp = torch.zeros(4, 8, 24, 32, dtype=torch.float64)
            exp[:, :, 0:4] = wpre[:, :, 0]
            exp[:, :, 4:8] = wpost[:, :, :, N]
            for r in range(N):
                exp[:, :, 8 + 4 * r : 12 + 4 * r] = wpost[:, :, r, 0:N]
            okp = torch.equal(Ph[:, :, 0, :24], exp) and torch.equal(Ph, Ph[:, :, :1].expand_as(Ph))
            if not okp:
                failures.append(f"{tag}: packed tile != release weights")
            _free([c, y, s, P])
            # trace: fused captured once, replayed with new inputs == eager ops
            def step(site):
                r_, c_ = site.pre(X)
                return [r_, site.post(X, O, c_)]

            _free(step(fus))
            ttnn.synchronize_device(mesh_device)
            with _Capture(mesh_device) as cap:
                tb = step(fus)
            try:
                for it in range(2):
                    x2, o2 = random_streams(4 * L, seed=200 + it + L)
                    xh2, oh2, kw2 = upload(x2, o2, L)
                    ttnn.copy_host_to_device_tensor(ttnn.from_torch(xh2, **kw2), X)
                    ttnn.copy_host_to_device_tensor(ttnn.from_torch(oh2, **kw2), O)
                    ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                    e = step(ops)
                    same = all(torch.equal(_chips(p, mesh_device), _chips(q, mesh_device)) for p, q in zip(tb, e))
                    print(f"[mhc] decode fused {tag} trace replay it{it}: == eager ops bitwise: {same}", flush=True)
                    if not same:
                        failures.append(f"{tag}: trace replay it{it} differs from eager ops")
                    _free(e)
            finally:
                ttnn.release_trace(mesh_device, cap.tid)
            _free(tb)
            if regime == "realistic":
                lat = {k: _traced_us(mesh_device, lambda: step(st), n=16) for k, st in (("ops", ops), ("fused", fus))}
                print(f"[mhc] LATENCY decode site T={L} {json.dumps({k: round(v, 1) for k, v in lat.items()})}",
                      flush=True)
            _free([X, O])
        ops.release()
        fus.release()

    # real weights and streams (MHC-5 decode sites): fused == ops bitwise
    path = DATA / "real_decode_tokens.pt"
    src = W.HFWeightLoader()
    n_real = 0
    if path.is_file():
        data = torch.load(path, weights_only=True)
        for key, d in sorted(data["sites"].items()):
            prompt, layer, site = key.split("|")
            layer = int(layer)
            if layer not in (0, 1, 30, 31) or not src.layer_available(layer):
                continue
            ops = M.MHCSite(mesh_device, cfg, layer, site, source=src, cache=False, decode="ops")
            fus = M.MHCSite(mesh_device, cfg, layer, site, source=src, cache=False, decode="fused")
            xh, oh, kw = upload(d["x"], d["out"], 8)
            X = ttnn.from_torch(xh, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
            O = ttnn.from_torch(oh, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
            compare(f"real/{prompt}/L{layer}/{site}", ops, fus, X, O)
            n_real += 1
            _free([X, O])
            ops.release()
            fus.release()
    print(f"[mhc] decode fused: {n_real} real sites compared", flush=True)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_prefill_latency(mesh_device):
    """Prefill buckets S = 128 ... 32768 (random weights, replicated streams): eager latency per stage after a warm-up
    call (the first call per shape JIT-compiles), the kernel finalize / concat-free post against their op forms
    (bitwise post, <= 1e-6 p), and accuracy on 256 sampled tokens (mHC is per token)."""
    from models.demos.motif3.tt import mhc as M

    log_fabric(mesh_device, "test_mhc_prefill_latency")
    cfg = _cfg(mesh_device)
    src = random_site_source(4, "mhc_ffn", seed=5)
    proj, gamma, sc = site_tensors(src, W.hf_name(4, "mhc_ffn"))
    m64 = ref_layer(proj, gamma, sc)
    failures = []
    for backend in BACKENDS:
        st = M.MHCSite(mesh_device, cfg, 4, "mhc_ffn", source=src, cache=False, sinkhorn=backend)
        st_ops = M.MHCSite(mesh_device, cfg, 4, "mhc_ffn", source=src, cache=False, sinkhorn=backend, finalize="ttnn",
                           post_concat=True) if backend == "motif" else None
        for S in (128, 1024, 4096, 8192, 32768):
            x, o = random_streams(S, seed=S)
            X = _replicated(mesh_device, to_tt_layout(x))
            O = _replicated(mesh_device, o[None, None])
            t = {}

            def timed(name, fn):
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                r = fn()
                ttnn.synchronize_device(mesh_device)
                t[name] = (time.perf_counter() - t0) * 1e3
                return r

            for _ in range(2):  # warm-up: compile every program of the site (pre, post, keep=True path)
                xr0, c0 = st.pre(X, keep=True)
                _free([xr0, st.post(X, O, c0)])
            ttnn.synchronize_device(mesh_device)
            p = timed("mixes", lambda: st.mixes(X))
            maps = timed("maps", lambda: st.maps(p, halve_post=st.glue != "kernel"))
            c = timed("coefficients", lambda: st.coefficients(X, keep=True))
            xr = timed("apply_pre", lambda: st.apply_pre(X, c))
            xn = timed("apply_post", lambda: st.apply_post(X, O, c))
            tot = timed("site_total", lambda: [st.post(X, O, st.pre(X)[1]), None])
            t["glue"] = t["coefficients"] - t["mixes"] - t["maps"]
            extra = {}
            if st_ops is not None and S in (4096, 32768):  # op forms: finalize="ttnn" mixes, stock op on the concat
                _free([st_ops.mixes(X), st_ops.apply_post(X, O, c)])  # warm-up
                p2 = timed("mixes_ttnn_finalize", lambda: st_ops.mixes(X))
                xn2 = timed("apply_post_concat", lambda: st_ops.apply_post(X, O, c))
                d0 = lambda tt_: ttnn.to_torch(ttnn.get_device_tensors(tt_)[0]).double()  # noqa: E731
                pa, pb = d0(p)[0, 0, :, :24], d0(p2)[0, 0, :, :24]
                extra["p_kernel_vs_ops_rel"] = float((pa - pb).abs().max() / pb.abs().max())
                extra["post_fused_eq_concat"] = bool(torch.equal(d0(xn), d0(xn2)))
                if extra["p_kernel_vs_ops_rel"] > 1e-6 or not extra["post_fused_eq_concat"]:
                    failures.append(f"prefill/{backend} S={S}: kernel vs op forms {extra}")
                _free([p2, xn2])
            # accuracy on 256 sampled tokens
            idx = torch.linspace(0, S - 1, 256).long()
            g = golden(m64, x[idx], o[idx])
            d0 = lambda tt_: ttnn.to_torch(ttnn.get_device_tensors(tt_)[0]).double()  # noqa: E731
            hp, hq, Hd = d0(c.h_pre)[idx, :4], d0(c.h_post)[idx, :4], d0(c.H)[idx, :16]
            xr_s, xn_s = d0(xr)[0, 0, idx], d0(xn)[0][:, idx]
            e = dict(
                H=_maxabs(Hd, g["H"]),
                h_pre=_maxabs(hp, g["h_pre"]),
                h_post=_maxabs(hq, g["h_post"]),
                p_rel=float((d0(c.p)[0, 0, idx, :24] - g["p"]).norm() / g["p"].norm()),
                x_red_pcc=_pcc(xr_s, g["x_red"]),
                x_out_pcc=_pcc(xn_s, g["x_out"].permute(1, 0, 2)),
                replicas_identical_all=all(_replicas_all(tt_, mesh_device) for tt_ in (xr, xn)),
            )
            _merge_stage(e, mix_errors(x[idx].permute(1, 0, 2), o[idx], hp, hq, Hd, xr_s, xn_s),
                         maps_errors(d0(c.p)[0, 0, idx, :24], sc, hp, hq, Hd))
            e.update(extra)
            ok = _ok(e, backend, kind="random")
            line = (f"{'PASS' if ok else 'FAIL'} prefill/{backend} S={S}: {_fmt(e)} | eager ms "
                    + json.dumps({k: round(v, 2) for k, v in t.items()}))
            print("[mhc] " + line, flush=True)
            if not ok:
                failures.append(line)
            _free([p, list(maps), c, xr, xn, tot, X, O])
        st.release()
        if st_ops is not None:
            st_ops.release()
    assert not failures, "\n".join(failures)


# =====================================================================================================================
# opt-in probes (MOTIF3_MHC_PROBE=1): the measurements behind the design choices in tt/mhc.py
# =====================================================================================================================
def _stats(name, dev, ref):
    dev = dev.double().reshape(ref.shape)
    ref = ref.double()
    d = dev - ref
    out = dict(
        rel_rms=float(d.norm() / ref.norm().clamp(min=1e-30)),
        bias=float((d * ref.sign()).sum() / ref.abs().sum().clamp(min=1e-30)),
        max_abs=float(d.abs().max()),
    )
    print(f"[mhc.probe] {name}: " + " ".join(f"{k}={v:.3e}" for k, v in out.items()), flush=True)
    return out


def _bmm_pc(grid, kt, per_core_m=1, in0_block_w=None, per_core_n=1):
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(int(grid[0]), int(grid[1])),
        in0_block_w=int(in0_block_w or kt),
        out_subblock_h=1,
        out_subblock_w=int(per_core_n),
        per_core_M=int(per_core_m),
        per_core_N=int(per_core_n),
    )


def _probe_ckc():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


@pytest.mark.skipif(os.environ.get("MOTIF3_MHC_PROBE", "0") == "0", reason="opt-in probe (MOTIF3_MHC_PROBE=1)")
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_probe_numerics(mesh_device):
    """FPU / SFPU precision of the building blocks (recorded in tt/mhc.py's docstring): bf16 matmul -> fp32 ~3.1e-4
    rel RMS (intra-tile adder; not the products); dest accumulation of TF32-exact values exact; fast_reduce_nc fp32
    -3.5e-4 biased; ttnn.sum fp32 ~1e-7; bf16 x bf16 -> fp32 multiply exact; rsqrt fp32 ~3e-8."""
    log_fabric(mesh_device, "test_mhc_probe_numerics")
    g = torch.Generator().manual_seed(0)
    ckc = _probe_ckc()
    d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()  # noqa: E731
    X = torch.randn(1, 1, 32, 4096, generator=g).bfloat16()
    Wt = (torch.randn(1, 1, 4096, 32, generator=g) / 64).bfloat16()
    Xt, Wtt = _replicated(mesh_device, X), _replicated(mesh_device, Wt)
    _stats(
        "matmul bf16 -> fp32",
        d0(ttnn.matmul(Xt, Wtt, dtype=ttnn.float32, compute_kernel_config=ckc)),
        X.double() @ Wt.double(),
    )
    A = torch.rand(1, 1, 32, 1024, generator=g) + 0.5
    A = (A.view(torch.int32) & ~((1 << 13) - 1)).view(torch.float32)  # TF32-exact
    S = torch.zeros(1, 1, 1024, 32)
    for b in range(32):
        S[0, 0, b * 32 : (b + 1) * 32, :] = torch.eye(32)
    _stats(
        "matmul tf32-exact x 0/1 (dest accumulation)",
        d0(
            ttnn.matmul(
                _replicated(mesh_device, A, ttnn.float32),
                _replicated(mesh_device, S, ttnn.float32),
                dtype=ttnn.float32,
                compute_kernel_config=ckc,
            )
        ),
        A.double() @ S.double(),
    )
    P = torch.randn(1, 32, 32, 32, generator=g)
    Pt = _replicated(mesh_device, P, ttnn.float32)
    _stats(
        "fast_reduce_nc fp32 dim 1",
        d0(ttnn.experimental.fast_reduce_nc(Pt, dims=[1], output=None, compute_kernel_config=ckc)),
        P.double().sum(1, keepdim=True),
    )
    _stats(
        "ttnn.sum fp32 dim 1",
        d0(ttnn.sum(Pt, dim=1, keepdim=True, compute_kernel_config=ckc)),
        P.double().sum(1, keepdim=True),
    )
    Xs = torch.randn(1, 4, 32, 4096, generator=g).bfloat16()
    Xst = _replicated(mesh_device, Xs)
    _stats("multiply bf16 x bf16 -> fp32", d0(ttnn.multiply(Xst, Xst, dtype=ttnn.float32)), Xs.double() ** 2)
    pos = torch.rand(1, 1, 32, 32, generator=g) * 3 + 0.1
    _stats("rsqrt fp32", d0(ttnn.rsqrt(_replicated(mesh_device, pos, ttnn.float32))), pos.double().rsqrt())


@pytest.mark.skipif(os.environ.get("MOTIF3_MHC_PROBE", "0") == "0", reason="opt-in probe (MOTIF3_MHC_PROBE=1)")
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_probe_configs(mesh_device):
    """Projection program configs and sum-of-squares variants (decode shape): the auto config takes 63-229 us
    (in0_block_w 1); explicit one-block split-K configs 6.7-9.5 us; K-blocked configs lose precision (reload)."""
    log_fabric(mesh_device, "test_mhc_probe_configs")
    g = torch.Generator().manual_seed(1)
    ckc = _probe_ckc()
    grid = mesh_device.compute_with_storage_grid_size()
    grid = (grid.x, grid.y)
    d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()  # noqa: E731
    T = 8
    Xd = torch.randn(1, 4, T, 4096, generator=g).bfloat16()
    fn = (torch.randn(24, 16384, generator=g) / 128).bfloat16()
    blocks = torch.zeros(1, 4, 4096, 32)
    blocks[..., :24] = fn.float().reshape(24, 4, 4096).permute(1, 2, 0)
    p_ref = Xd.double().permute(0, 2, 1, 3).reshape(T, 16384) @ fn.double().T
    ss_ref = (Xd.double() ** 2).sum(dim=(1, 3))
    Xdt = _replicated(mesh_device, Xd)
    Bt = _replicated(mesh_device, blocks.bfloat16())
    X32 = ttnn.reshape(Xdt, ttnn.Shape([1, 4, 32, 4096]))
    assert X32.buffer_address() == Xdt.buffer_address()
    for nb in (16, 32, 64, 128):
        kp = 16384 // nb

        def mm(nb=nb, kp=kp):
            return ttnn.matmul(
                ttnn.experimental.view(X32, [1, nb, 32, kp]),
                ttnn.experimental.view(Bt, [1, nb, kp, 32]),
                dtype=ttnn.float32,
                compute_kernel_config=ckc,
                program_config=_bmm_pc(grid, kp // 32),
            )

        _stats(f"split-K NB={nb}", d0(mm())[0, :, :T, :24].sum(0), p_ref)
        print(f"[mhc.probe] latency split-K NB={nb}: {_traced_us(mesh_device, mm):.1f} us", flush=True)
    for ibw in (32, 64, 128):
        try:
            y = ttnn.matmul(
                Xdt, Bt, dtype=ttnn.float32, compute_kernel_config=ckc, program_config=_bmm_pc(grid, 128, in0_block_w=ibw)
            )
            _stats(f"per-stream in0_block_w={ibw}", d0(y)[0, :, :T, :24].sum(0), p_ref)
        except Exception as ex:
            print(f"[mhc.probe] per-stream in0_block_w={ibw}: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
    xv = ttnn.experimental.view(X32, [1, 32, 32, 512])
    s = ttnn.rms_norm_pre_all_gather(xv, dtype=ttnn.float32, compute_kernel_config=ckc)
    _stats("sum(x^2) rms_norm_pre_all_gather fp32", d0(s)[0, :, :T, 0].sum(0), ss_ref)
    s = ttnn.sum(ttnn.multiply(xv, xv, dtype=ttnn.float32), dim=-1, keepdim=True, compute_kernel_config=ckc)
    _stats("sum(x^2) exact multiply + sum", d0(s)[0, :, :T, 0].sum(0), ss_ref)


@pytest.mark.skipif(os.environ.get("MOTIF3_MHC_PROBE", "0") == "0", reason="opt-in probe (MOTIF3_MHC_PROBE=1)")
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_probe_prefill_proj(mesh_device):
    """Prefill projection [1,4,S,4096] @ [1,4,4096,64] (hi|lo): in0_block_w x packer_l1_acc precision / latency, and
    the decode split-K with N = 2 tiles."""
    log_fabric(mesh_device, "test_mhc_probe_prefill_proj")
    g = torch.Generator().manual_seed(2)
    grid = mesh_device.compute_with_storage_grid_size()
    grid = (grid.x, grid.y)
    d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()  # noqa: E731
    fn = (torch.randn(24, 16384, generator=g) / 128).bfloat16()
    blocks = torch.zeros(1, 4, 4096, 64)
    blocks[..., :24] = fn.float().reshape(24, 4, 4096).permute(1, 2, 0)
    Bt = _replicated(mesh_device, blocks.bfloat16())
    for S in (128, 1024):
        Xp = torch.randn(1, 4, S, 4096, generator=g).bfloat16()
        Xpt = _replicated(mesh_device, Xp)
        refp = Xp.double().permute(0, 2, 1, 3).reshape(S, 16384) @ fn.double().T
        for ibw in (32, 64, 128):
            for pl1 in (False, True):
                ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                              math_approx_mode=False, fp32_dest_acc_en=True,
                                                              packer_l1_acc=pl1)
                pc = _bmm_pc(grid, 128, in0_block_w=ibw, per_core_n=2)
                name = f"prefill S={S} ibw={ibw} packer_l1_acc={pl1}"
                try:
                    y = ttnn.matmul(Xpt, Bt, dtype=ttnn.float32, compute_kernel_config=ckc, program_config=pc)
                    _stats(name, d0(y)[0, :, :, :24].sum(0), refp)
                    ttnn.synchronize_device(mesh_device)
                    t0 = time.perf_counter()
                    for _ in range(5):
                        _free(ttnn.matmul(Xpt, Bt, dtype=ttnn.float32, compute_kernel_config=ckc, program_config=pc))
                    ttnn.synchronize_device(mesh_device)
                    print(f"[mhc.probe] latency {name}: eager {(time.perf_counter() - t0) / 5 * 1e6:.1f} us", flush=True)
                except Exception as ex:
                    print(f"[mhc.probe] {name}: {type(ex).__name__}: {str(ex)[:160]}", flush=True)
    T = 8
    Xd = torch.randn(1, 4, T, 4096, generator=g).bfloat16()
    Xdt = _replicated(mesh_device, Xd)
    X32 = ttnn.reshape(Xdt, ttnn.Shape([1, 4, 32, 4096]))
    refd = Xd.double().permute(0, 2, 1, 3).reshape(T, 16384) @ fn.double().T
    ckc = _probe_ckc()
    for nb in (16, 32, 64):
        kp = 16384 // nb

        def mm(nb=nb, kp=kp):
            return ttnn.matmul(ttnn.experimental.view(X32, [1, nb, 32, kp]), ttnn.experimental.view(Bt, [1, nb, kp, 64]),
                               dtype=ttnn.float32, compute_kernel_config=ckc,
                               program_config=_bmm_pc(grid, kp // 32, per_core_n=2))

        _stats(f"decode split-K N=2 NB={nb}", d0(mm())[0, :, :T, :24].sum(0), refd)
        print(f"[mhc.probe] latency decode split-K N=2 NB={nb}: {_traced_us(mesh_device, mm):.1f} us", flush=True)


@pytest.mark.skipif(os.environ.get("MOTIF3_MHC_PROBE", "0") == "0", reason="opt-in probe (MOTIF3_MHC_PROBE=1)")
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mhc_probe_wall(mesh_device):
    """Prefill / decode projection as one non-batched matmul X[4T, 4096] @ W_all[4096, 4 x 64] (auto mcast config,
    exact fp32 partial reload) + diagonal-block mask + exact sums."""
    log_fabric(mesh_device, "test_mhc_probe_wall")
    g = torch.Generator().manual_seed(3)
    ckc = _probe_ckc()
    d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()  # noqa: E731
    fn = (torch.randn(24, 16384, generator=g) / 128).bfloat16()
    wall = torch.zeros(4096, 256)
    for s in range(4):
        wall[:, 64 * s : 64 * s + 24] = fn.float()[:, 4096 * s : 4096 * (s + 1)].T
    Wt = _replicated(mesh_device, wall[None, None].bfloat16())
    mask = torch.zeros(1, 4, 1, 256)
    for s in range(4):
        mask[0, s, 0, 64 * s : 64 * (s + 1)] = 1.0
    Mt = _replicated(mesh_device, mask, ttnn.float32)

    def proj(Xt, S):
        St = S // 32
        y = ttnn.matmul(ttnn.reshape(Xt, [1, 1, 4 * S, 4096]), Wt, dtype=ttnn.float32, compute_kernel_config=ckc)
        y = ttnn.reshape(y, [1, 4, S, 256])
        ym = ttnn.multiply(y, Mt)
        y1 = ttnn.sum(ym, dim=1, keepdim=True, compute_kernel_config=ckc)  # [1, 1, S, 256]
        y8 = ttnn.experimental.view(y1, [St, 8, 32, 32])
        p = ttnn.sum(y8, dim=1, keepdim=True, compute_kernel_config=ckc)  # [St, 1, 32, 32]
        return ttnn.reshape(p, [1, 1, S, 32])

    for S in (32, 128, 1024, 4096):
        Xp = torch.randn(1, 4, S, 4096, generator=g).bfloat16()
        Xpt = _replicated(mesh_device, Xp)
        refp = Xp.double().permute(0, 2, 1, 3).reshape(S, 16384) @ fn.double().T
        try:
            got = d0(proj(Xpt, S))[0, 0, :, :24]
            _stats(f"W_all S={S}", got, refp)
            if S == 32:
                print(f"[mhc.probe] latency W_all decode-shape S=32: {_traced_us(mesh_device, lambda: proj(Xpt, 32)):.1f} us traced", flush=True)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(3):
                _free(proj(Xpt, S))
            ttnn.synchronize_device(mesh_device)
            print(f"[mhc.probe] latency W_all S={S}: eager {(time.perf_counter() - t0) / 3 * 1e6:.1f} us", flush=True)
            _free(ttnn.matmul(ttnn.reshape(Xpt, [1, 1, 4 * S, 4096]), Wt, dtype=ttnn.float32, compute_kernel_config=ckc))
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(3):
                _free(ttnn.matmul(ttnn.reshape(Xpt, [1, 1, 4 * S, 4096]), Wt, dtype=ttnn.float32, compute_kernel_config=ckc))
            ttnn.synchronize_device(mesh_device)
            print(f"[mhc.probe] latency W_all matmul only S={S}: eager {(time.perf_counter() - t0) / 3 * 1e6:.1f} us", flush=True)
        except Exception as ex:
            print(f"[mhc.probe] W_all S={S}: {type(ex).__name__}: {str(ex)[:300]}", flush=True)


# =====================================================================================================================
# host-only capture of the real-stream test data (no device; run through scripts/hostrun.sh)
# =====================================================================================================================
def capture_real_streams(out_dir=DATA, n_layers=36):
    """bf16 reference prefix model (layers 0..n_layers-1, lazy experts) on two chat prompts (the reference default,
    145 tokens; a model-card summary, 2027 tokens). Writes, per decode-test site (layers 0-5, 28-35, both sites,
    both prompts), 32 tokens (16 with the largest emulated TF32-logit Sinkhorn error + 16 random): the site input
    ``x [32, 4, D]`` and the sublayer output ``out [32, D]``; full sequences for the default prompt (layers 0, 2, 4) and
    for the long prompt at layer 30 mhc_attn (the worst C3 site)."""
    from models.demos.motif3.reference.golden import DEFAULT_PROMPT_MESSAGES, capture_model_goldens
    from models.demos.motif3.reference.tokenizer import encode_chat, load_tokenizer
    from models.demos.motif3.reference.weights import MotifCheckpoint, load_reference_model

    torch.manual_seed(0)
    os.makedirs(out_dir, exist_ok=True)
    model = load_reference_model(
        layer_ids=range(n_layers), dtype=torch.bfloat16, lazy_experts=True, checkpoint=MotifCheckpoint()
    )
    tok = load_tokenizer()
    card = (Path(HF_META) / "README.md").read_text()
    prompts = {
        "default": DEFAULT_PROMPT_MESSAGES,
        "card": [{"role": "user", "content": "Summarize the following model card in detail.\n\n" + card[:6000]}],
    }
    layers = [i for i in REAL_DECODE_LAYERS if i < n_layers]
    names = ("x_in", "x_mid", "self_attn.out", "mlp.out", "moe.out", "mhc_attn.mixes", "mhc_ffn.mixes")
    want = {f"layers.{i}.{n}" for i in layers for n in names}

    def sinkhorn(L):
        m = L.double().clamp(-20.0, 20.0).exp()
        for _ in range(20):
            m = m / m.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            m = m / m.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        return m

    def tf32(x):
        return (x.float().contiguous().view(torch.int32) & ~((1 << 13) - 1)).view(torch.float32)

    sites = {}
    meta = {"prompts": {}, "layers": layers, "dtype": "bfloat16"}
    for pname, msgs in prompts.items():
        ids = torch.tensor([encode_chat(msgs, tok)])
        meta["prompts"][pname] = int(ids.shape[1])
        g = capture_model_goldens(model, ids, None, include=lambda n: n in want)["prefill"]
        S = ids.shape[1]
        for i in layers:
            layer = model.model.layers[str(i)]
            ffn_out = g[f"layers.{i}.moe.out"] if layer.is_moe else g[f"layers.{i}.mlp.out"]
            for site, xk, out in (("mhc_attn", "x_in", g[f"layers.{i}.self_attn.out"]), ("mhc_ffn", "x_mid", ffn_out)):
                x = g[f"layers.{i}.{xk}"][0]
                m = getattr(layer, site)
                p = g[f"layers.{i}.{site}.mixes"][0].double().reshape(S, 24)
                L = (float(m.alpha_res.double()) * p[:, 8:] + m.bias_res.double().reshape(-1)).reshape(S, 4, 4)
                score = (sinkhorn(tf32(L)) - sinkhorn(L)).abs().amax(dim=(1, 2))
                hard = torch.argsort(score, descending=True)[:16]
                rest = torch.tensor([t for t in torch.randperm(S).tolist() if t not in set(hard.tolist())][:16])
                idx = torch.cat([hard, rest]).sort().values
                sites[f"{pname}|{i}|{site}"] = {
                    "x": x[idx].clone(),
                    "out": out[0][idx].clone(),
                    "tok": idx.clone(),
                    "hard_score_max": float(score.max()),
                }
            if pname == "default" and i in (0, 2, 4):
                torch.save(
                    {
                        "x_in": g[f"layers.{i}.x_in"][0].clone(),
                        "attn_out": g[f"layers.{i}.self_attn.out"][0].clone(),
                        "x_mid": g[f"layers.{i}.x_mid"][0].clone(),
                        "ffn_out": ffn_out[0].clone(),
                        "layer": i,
                    },
                    Path(out_dir) / f"real_prefill_default_L{i:02d}.pt",
                )
            if pname == "card" and i == 30:
                torch.save(
                    {
                        "x": g[f"layers.{i}.x_in"][0].clone(),
                        "out": g[f"layers.{i}.self_attn.out"][0].clone(),
                        "layer": i,
                        "site": "mhc_attn",
                    },
                    Path(out_dir) / "real_prefill_card_L30_mhc_attn.pt",
                )
        del g
    torch.save({"meta": meta, "sites": sites}, Path(out_dir) / "real_decode_tokens.pt")
    return sorted(os.listdir(out_dir))


if __name__ == "__main__":
    if "--capture" in sys.argv:
        print(json.dumps({"files": capture_real_streams()}))
