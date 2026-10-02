# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight decoder-layer tests vs the C2 goldens (WAVE_A_REVIEW GEN-7(a); README §12: decoder layer >= 0.995,
prefill and teacher-forced decode).

For each layer K in ``LAYERS`` (dense global 0, dense SWA 1, MoE SWA 2 / 3, MoE global 4 / 8 / 16 / 24 / 32, and the
mHC review's worst sites 28-35, GEN-7(a) / MHC-5) the layer input is the golden 4-stream residual after layer K-1
(layer 0: the golden embedding, 4 identical streams) and the target the golden residual after layer K (``goldens/c2``:
6 real prompts, 2971 tokens, the reference's bf16 HF numerics; ``reference/golden_stream.py``).

The C2 run saved the states after layers {0-4, 7, 8, 15, 16, 23, 24, 31, 32, 35} only. The states after 25-30 and 33-34
(``EXT_DIR``) are the same reference run continued on the CPU from the saved golden after 24 / 32
(``golden_stream.run_stream(resume=True)``: the reference modules exactly as the C2 run called them, same numerics,
thread-count independent); :func:`make_extension_goldens` accepts them only if the continued run reproduces the saved
goldens after 31 and 35 **bitwise**. Build them once (host, ~1 min)::

    scripts/hostrun.sh -t 1800 -- python models/demos/motif3/tests/test_decoder_layer.py --make-goldens

* **prefill**: every prompt as one user (bucket-padded with real, non-zero rows of another prompt), no cache fill;
  PCC of the 4-stream state on the real positions, per prompt.
* **teacher-forced decode**: 32 lanes (4 DP rows x 8, two lanes inactive), each lane a prompt prefix ``0 .. q0 - 1``
  prefilled into its own (shuffled) blocks of one paged cache, then ``DECODE_STEPS`` decode steps at positions
  ``q0 .. q0 + 15`` fed with the golden inputs of those positions; heterogeneous positions per lane (the SWA window
  is crossed: q0 down to 109). PCC per lane over its 16 positions; TP replicas must be bitwise identical.
* the traced decode latency of the layer call (slope method), for the per-layer cost model.

Reported per test: state PCC (the acceptance metric), the PCC of the layer's update ``X_out - X_in`` (a much stricter
view: the residual dominates the state), max-abs error and non-finite counts.

Run (device, through the lock wrapper; ~1-1.5 min per layer)::

    scripts/devrun.sh -t 2400 -n decoder_layer -- python -m pytest models/demos/motif3/tests/test_decoder_layer.py \
        -s -p no:cacheprovider --timeout=0
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.motif3.tt.model_config import device_params

GOLDEN_DIR = Path(os.environ.get("MOTIF3_GOLDEN_STREAM_DIR", "/home/ttuser/hchang/experiments/motif-3/goldens/c2"))
EXT_DIR = Path(os.environ.get("MOTIF3_GOLDEN_EXT_DIR",
                              "/home/ttuser/hchang/experiments/motif-3/tt_cache/test/integration/goldens_c2_ext"))
EXT_SPANS = ((24, 31), (32, 35))  # (saved golden to continue from, next saved golden = the bitwise check)
LAYERS = tuple(int(x) for x in os.environ.get(
    "MOTIF3_TEST_LAYERS", "0,1,2,3,4,8,16,24,28,29,30,31,32,33,34,35").split(","))
DECODE_STEPS = 16
PCC_MIN = 0.995  # README §12 / design §4.3: decoder layer
MAX_MODEL_LEN = 2048  # covers the longest golden prompt (893 tokens -> bucket 1024)
NUM_BLOCKS = 512
BLOCK = 64
INACTIVE_LANES = (3, 30)
MESH = [pytest.param((4, 8), device_params(), id="4x8")]


# ============================================================================================================
# host helpers
# ============================================================================================================
def stats(ref: torch.Tensor, got: torch.Tensor) -> dict:
    """PCC (``comp_pcc`` in float64, README §12) + max-abs + non-finite count."""
    from models.common.utility_functions import comp_pcc

    r, g = ref.double(), got.double()
    nonfinite = int((~torch.isfinite(g)).sum())
    _, pcc = comp_pcc(r, g, 0.0)
    return {"pcc": float(pcc) if nonfinite == 0 else float("nan"), "max_abs": float((r - g).abs().max()),
            "nonfinite": nonfinite}


def fmt(s: dict) -> str:
    return f"pcc={s['pcc']:.6f} max_abs={s['max_abs']:.3e}" + (f" NONFINITE={s['nonfinite']}" if s["nonfinite"] else "")


def _state_file(l):
    """Golden state file after layer ``l`` (the C2 run, else its CPU continuation in ``EXT_DIR``); ``l = -1`` = the
    embedding (the layer-0 input is 4 copies of it)."""
    from models.demos.motif3.reference import golden_stream as gs

    if l < 0:
        return GOLDEN_DIR / "states" / "embed.safetensors"
    p = gs.state_path(GOLDEN_DIR, l)
    return p if p.is_file() else gs.state_path(EXT_DIR, l)


def make_extension_goldens(out_dir: Path = EXT_DIR, spans=EXT_SPANS, threads: int = 32) -> dict:
    """Host: the C2 states after the layers between saved goldens, by continuing the reference run on the CPU from
    the saved golden after ``start`` through ``end`` (``golden_stream.run_stream(resume=True)`` on a copy of the run's
    manifest / prompts with the state after ``start`` as its resume checkpoint). The recomputed state after ``end`` must
    equal the saved golden bitwise (else nothing is kept); the states after ``start + 1 .. end - 1`` go to
    ``out_dir/states``. Returns ``{span: {"layers": [...], "bitwise": bool}}``."""
    import json
    import shutil

    from models.demos.motif3.reference import golden_stream as gs

    out_dir = Path(out_dir)
    (out_dir / "states").mkdir(parents=True, exist_ok=True)
    report = {}
    for start, end in spans:
        work = out_dir / f".work_from_L{start:02d}"
        shutil.rmtree(work, ignore_errors=True)
        (work / "resume").mkdir(parents=True)
        try:
            shutil.copy(GOLDEN_DIR / "prompts.json", work / "prompts.json")
            shutil.copy(gs.state_path(GOLDEN_DIR, start), work / "resume" / "state.safetensors")
            man = json.loads((GOLDEN_DIR / "manifest.json").read_text())
            man.update(last_layer=start, layers_done=list(range(start + 1)), heads={}, runs=[], files={},
                       saved_layers=[], layer_log=[e for e in man.get("layer_log", []) if e["layer_idx"] <= start])
            (work / "manifest.json").write_text(json.dumps(man))
            gs.run_stream(work, layers=list(range(start + 1, end + 1)), resume=True,
                          save_layers=list(range(start + 1, end + 1)), final_head=False, threads=threads)
            got, _ = gs.load_tensors(gs.state_path(work, end))
            want, _ = gs.load_tensors(gs.state_path(GOLDEN_DIR, end))
            same = set(got) == set(want) and all(torch.equal(got[k], want[k]) for k in want)
            print(f"[ext goldens] layers {start + 1}..{end} continued from the golden after {start}: recomputed state "
                  f"after {end} bitwise equal to the saved golden: {same}")
            if not same:
                raise AssertionError(f"the CPU continuation from layer {start} does not reproduce the golden after {end}")
            for l in range(start + 1, end):
                shutil.move(str(gs.state_path(work, l)), str(gs.state_path(out_dir, l)))
            report[f"{start}->{end}"] = {"layers": list(range(start + 1, end)), "bitwise": same}
        finally:
            shutil.rmtree(work, ignore_errors=True)
    (out_dir / "ext_manifest.json").write_text(json.dumps(
        {"source": str(GOLDEN_DIR), "method": "golden_stream.run_stream(resume=True) from the saved golden after "
         "each span start; kept only if the state after the span end equals the saved golden bitwise",
         "threads": threads, "spans": report}, indent=1))
    return report


def load_goldens(layers):
    """``({layer: {prompt: [S, 4, 4096] bf16}}, [prompt names in order])`` from the C2 golden-stream run (layer -1:
    the embedding expanded to the 4 identical streams, i.e. the layer-0 input)."""
    from models.demos.motif3.reference import golden_stream as gs

    pfile = GOLDEN_DIR / "prompts.json"
    missing = [str(p) for p in [pfile] + [_state_file(l) for l in layers] if not p.is_file()]
    if missing:
        pytest.skip(f"C2 golden-stream files missing: {missing} (states between saved goldens: run this file with "
                    f"--make-goldens, see the module docstring)")
    prompts = gs.load_prompt_set(pfile)
    sha = gs.prompt_set_sha256(prompts)
    out = {}
    for l in layers:
        states, meta = gs.load_tensors(_state_file(l))
        assert meta.get("prompt_sha256", sha) == sha and meta.get("mode", {}).get("dtype", "bf16") == "bf16", (
            l, meta.get("mode"))
        if l < 0:
            out[l] = {k: v[0].unsqueeze(1).expand(-1, 4, -1).contiguous() for k, v in states.items()}  # [S, 4, D]
        else:
            out[l] = {k: v[0] for k, v in states.items()}
    return out, [p.name for p in prompts]


def real_source_or_skip(layers):
    from models.demos.motif3.tt.weights import HFWeightLoader

    try:
        src = HFWeightLoader()
    except FileNotFoundError as e:
        pytest.skip(f"no local checkpoint: {e}")
    missing = [l for l in layers if not src.layer_available(l)]
    if missing:
        pytest.skip(f"checkpoint layers {missing} are not local")
    return src


def bucket_of(S: int) -> int:
    b = 128
    while b < S:
        b *= 2
    return b


# ============================================================================================================
# device helpers
# ============================================================================================================
def _free(*ts):
    for t in ts:
        if t is None:
            continue
        if isinstance(t, (list, tuple)):
            _free(*t)
        elif isinstance(t, dict):
            _free(*t.values())
        elif t.is_allocated():
            ttnn.deallocate(t)


def upload_streams(x: torch.Tensor, mesh_device):
    """``[S, 4, 4096]`` (reference token-major) -> replicated ``[1, 4, S, 4096]`` bf16 TILE (stream-major)."""
    t = x.permute(1, 0, 2).unsqueeze(0).contiguous().to(torch.bfloat16)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))


def read_streams(t, S: int) -> torch.Tensor:
    """Replicated ``[1, 4, S', 4096]`` -> chip 0's ``[S, 4, 4096]`` (token-major)."""
    v = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return v[0, :, :S].permute(1, 0, 2).contiguous()


def upload_lane_streams(x32: torch.Tensor, cfg, mesh_device):
    """Lane-ordered ``[32, 4, 4096]`` -> per DP row ``[1, 4, 8, 4096]`` (row r = lanes 8r .. 8r+7)."""
    from models.demos.motif3.tt.rope import shard_lanes

    rows = x32.reshape(cfg.dp, cfg.lanes_per_row, 4, -1).permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    return shard_lanes(rows, cfg, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)


def read_lane_streams(t, cfg, mesh_device):
    """Per-row ``[1, 4, 8, 4096]`` -> lane-ordered ``[32, 4, 4096]`` (chip tp = 0 of each row) + TP replicas identical."""
    from models.demos.motif3.tt.ccl import device_tensors_to_torch

    full = device_tensors_to_torch(t, mesh_device)  # [R, C, 1, 4, 8, 4096]
    out = []
    same = True
    for dp in range(cfg.dp):
        r, c = cfg.axes.coord(dp, 0)
        out.append(full[r, c, 0].permute(1, 0, 2))  # [8, 4, 4096]
        for tp in range(cfg.tp):
            rr, cc = cfg.axes.coord(dp, tp)
            same &= torch.equal(full[rr, cc], full[r, c])
    return torch.cat(out, 0).float(), same


class _Capture:
    """Exception-safe trace capture (README §0: a dangling capture hung close_mesh_device once)."""

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


def traced_us(mesh_device, fn, n: int = 8, reps: int = 5) -> dict:
    """Traced us per ``fn()`` call by the gates' slope method: ``(t(n) - t(n/2)) / (n/2)``, min over replays."""

    def t_of(k):
        with _Capture(mesh_device) as cap:
            for _ in range(k):
                _free(fn())
        try:
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                ts.append((time.perf_counter() - t0) * 1e6)
        finally:
            ttnn.release_trace(mesh_device, cap.tid)
        return min(ts)

    t1, t2 = t_of(n // 2), t_of(n)
    return {"slope_us": (t2 - t1) / (n - n // 2), "raw_us": t2 / n}


# ============================================================================================================
# the test
# ============================================================================================================
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("layer", LAYERS, ids=[f"L{l}" for l in LAYERS])
@torch.no_grad()
def test_decoder_layer_real(mesh_device, device_params, layer):
    from models.demos.motif3.tt.attention import MotifAttention
    from models.demos.motif3.tt.ccl import MotifCCL, log_fabric
    from models.demos.motif3.tt.decoder import MotifDecoderLayer
    from models.demos.motif3.tt.model_config import MotifTTConfig
    from models.demos.motif3.tt.rope import MotifRope, shard_lanes

    K = int(layer)
    goldens, names = load_goldens((K - 1, K))
    src = real_source_or_skip((K,))
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device, max_model_len=MAX_MODEL_LEN, num_layers=53)
    cfg.set_kv_geometry(NUM_BLOCKS, BLOCK)
    log_fabric(mesh_device, f"decoder_layer L{K}")
    print(f"[decoder L{K}] {cfg.describe()}")
    ccl = MotifCCL(mesh_device, cfg)
    rope = MotifRope(mesh_device, cfg)
    t0 = time.time()
    dec = MotifDecoderLayer(mesh_device, cfg, K, source=src, ccl=ccl, rope=rope, cache=False)
    t_load = time.time() - t0
    spec = cfg.layer(K)
    print(f"[decoder L{K}] {spec.kind} (window {spec.window}, rope {spec.rope_kind}) loaded in {t_load:.1f} s; "
          f"sinkhorn={cfg.mhc_sinkhorn} router={cfg.router_logits}")
    gin, gout = goldens[K - 1], goldens[K]
    failures = []

    # ---- prefill: every prompt as one user ------------------------------------------------------------------------
    t0 = time.time()
    pre_all_ref, pre_all_got, pre_all_d_ref, pre_all_d_got = [], [], [], []
    for i, name in enumerate(names):
        x, y = gin[name], gout[name]
        S = x.shape[0]
        B = bucket_of(S)
        filler = gin[names[(i + 1) % len(names)]]  # real, non-zero rows in the bucket padding
        pad = torch.cat([filler] * math.ceil((B - S) / filler.shape[0] + 1))[: B - S] if B > S else x[:0]
        X = upload_streams(torch.cat([x, pad]), mesh_device)
        out = dec.forward_prefill(X)
        got = read_streams(out, S)
        _free(X, out)
        s = stats(y, got)
        sd = stats(y.float() - x.float(), got - x.float())
        print(f"[decoder L{K}] prefill {name:18s} S={S:4d} (bucket {B}): state {fmt(s)}; update X_out-X_in {fmt(sd)}")
        if not s["pcc"] >= PCC_MIN:
            failures.append(f"prefill {name}: {fmt(s)}")
        pre_all_ref.append(y.float())
        pre_all_got.append(got)
        pre_all_d_ref.append(y.float() - x.float())
        pre_all_d_got.append(got - x.float())
    s_all = stats(torch.cat(pre_all_ref), torch.cat(pre_all_got))
    sd_all = stats(torch.cat(pre_all_d_ref), torch.cat(pre_all_d_got))
    print(f"[decoder L{K}] prefill all {sum(r.shape[0] for r in pre_all_ref)} tokens: state {fmt(s_all)}; update "
          f"{fmt(sd_all)} ({time.time() - t0:.1f} s)")
    del pre_all_ref, pre_all_got, pre_all_d_ref, pre_all_d_got

    # ---- teacher-forced decode: 32 lanes, prefix prefill into each lane's blocks, then 16 steps ---------------------
    empty = ttnn.empty([NUM_BLOCKS, 1, BLOCK, cfg.kv_latent_dim], cfg.dtypes.kv_cache, ttnn.TILE_LAYOUT, mesh_device,
                       ttnn.DRAM_MEMORY_CONFIG)
    kv = ttnn.fill(empty, 0.0)
    ttnn.deallocate(empty)
    L, B32 = cfg.lanes_per_row, cfg.max_batch
    W = math.ceil(MAX_MODEL_LEN / BLOCK)
    lanes = {}
    perm = (torch.randperm(NUM_BLOCKS - 1, generator=torch.Generator().manual_seed(K)) + 1).tolist()
    nxt = 0
    for lane in range(B32):
        if lane in INACTIVE_LANES:
            continue
        name = names[lane % len(names)]
        S = gin[name].shape[0]
        q0 = S - DECODE_STEPS - 4 * (lane // len(names))
        nb = math.ceil((q0 + DECODE_STEPS) / BLOCK)
        blocks = perm[nxt: nxt + nb]
        nxt += nb
        lanes[lane] = dict(name=name, q0=q0, blocks=blocks)
    pt = torch.zeros(B32, W, dtype=torch.int32)
    t0 = time.time()
    for lane, d in lanes.items():
        q0 = d["q0"]
        Bk = bucket_of(q0)
        x = gin[d["name"]][:q0]
        filler = gin[names[(lane + 1) % len(names)]]
        pad = torch.cat([filler] * math.ceil((Bk - q0) / filler.shape[0] + 1))[: Bk - q0]
        X = upload_streams(torch.cat([x, pad]), mesh_device)
        n_pt = cfg.prefill_page_table_entries(Bk)
        own = math.ceil(q0 / BLOCK)
        ppt = torch.zeros(1, n_pt, dtype=torch.int32)
        ppt[0, :own] = torch.tensor(d["blocks"][:own], dtype=torch.int32)  # the bridge zeroes the tail (null block)
        ppt_tt = ttnn.from_torch(ppt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
        _free(dec.forward_prefill(X, page_table=ppt_tt, kv_cache=kv))
        _free(X, ppt_tt)
        pt[lane, : len(d["blocks"])] = torch.tensor(d["blocks"], dtype=torch.int32)
    print(f"[decoder L{K}] decode setup: {len(lanes)} lanes prefilled (prefixes {min(d['q0'] for d in lanes.values())}"
          f"..{max(d['q0'] for d in lanes.values())} tokens) in {time.time() - t0:.1f} s; {nxt} blocks of the "
          f"{NUM_BLOCKS}-block pool")

    def step_inputs(t):
        pos = torch.full((B32,), -1, dtype=torch.int32)
        x32 = torch.zeros(B32, 4, cfg.hidden_size)
        for lane, d in lanes.items():
            q = d["q0"] + t
            pos[lane] = q
            x32[lane] = gin[d["name"]][q].float()
        for lane in INACTIVE_LANES:  # finite, non-zero rows on inactive lanes (never compared)
            x32[lane] = gin[names[0]][lane].float()
        ptt = torch.where((pos >= 0)[:, None], pt, torch.zeros_like(pt))
        dev = {
            "x": upload_lane_streams(x32, cfg, mesh_device),
            "cur": shard_lanes(pos, cfg, mesh_device, dtype=ttnn.int32, device=mesh_device),
            "pt": shard_lanes(ptt, cfg, mesh_device, dtype=ttnn.int32, device=mesh_device),
            "rot_idx": rope.rot_idxs_device(pos),
        }
        dev["rot"] = MotifAttention.decode_rope_tables(rope, dev["rot_idx"])
        dev["act"] = MotifAttention.active_mask_from_cur_pos(dev["cur"], L)
        return dev, x32

    got_lane = {lane: [] for lane in lanes}
    in_lane = {lane: [] for lane in lanes}
    replicas = True
    t0 = time.time()
    dev = None
    for t in range(DECODE_STEPS):
        dev, x32 = step_inputs(t)
        out = dec.forward_decode(dev["x"], rot=dev["rot"], cur_pos=dev["cur"], page_table=dev["pt"], kv_cache=kv,
                                 active=dev["act"])
        got, same = read_lane_streams(out, cfg, mesh_device)
        replicas &= same
        _free(out)
        for lane in lanes:
            got_lane[lane].append(got[lane])
            in_lane[lane].append(x32[lane])
        if t < DECODE_STEPS - 1:
            _free(dev)
    t_dec = time.time() - t0
    worst = (2.0, None)
    all_ref, all_got, all_dr, all_dg = [], [], [], []
    for lane, d in lanes.items():
        q0 = d["q0"]
        ref = gout[d["name"]][q0: q0 + DECODE_STEPS].float()
        got = torch.stack(got_lane[lane])
        xin = torch.stack(in_lane[lane])
        s = stats(ref, got)
        if s["pcc"] < worst[0] or s["pcc"] != s["pcc"]:
            worst = (s["pcc"], lane)
        if not s["pcc"] >= PCC_MIN:
            failures.append(f"decode lane {lane} ({d['name']} q {q0}..{q0 + DECODE_STEPS - 1}): {fmt(s)}")
        all_ref.append(ref)
        all_got.append(got)
        all_dr.append(ref - xin)
        all_dg.append(got - xin)
    s_dec = stats(torch.cat(all_ref), torch.cat(all_got))
    sd_dec = stats(torch.cat(all_dr), torch.cat(all_dg))
    print(f"[decoder L{K}] decode {len(lanes)} lanes x {DECODE_STEPS} steps (positions "
          f"{min(d['q0'] for d in lanes.values())}..{max(d['q0'] for d in lanes.values()) + DECODE_STEPS - 1}): state "
          f"{fmt(s_dec)}; update {fmt(sd_dec)}; worst lane {worst[1]} pcc={worst[0]:.6f}; TP replicas identical "
          f"{replicas}; eager {t_dec / DECODE_STEPS * 1e3:.1f} ms per step incl. host I/O")
    if not replicas:
        failures.append("decode outputs differ between the TP replicas of a row")

    # ---- traced decode latency of this layer (the last step's inputs; replays rewrite the same KV slots) -------------
    try:
        lat = traced_us(mesh_device, lambda: dec.forward_decode(dev["x"], rot=dev["rot"], cur_pos=dev["cur"],
                                                                 page_table=dev["pt"], kv_cache=kv, active=dev["act"]))
        print(f"[decoder L{K}] traced decode per layer call ({spec.kind}, contexts up to "
              f"{max(d['q0'] for d in lanes.values()) + DECODE_STEPS}): {lat['slope_us']:.1f} us (slope), "
              f"{lat['raw_us']:.1f} us raw")
    except Exception as e:  # latency is a report, not an acceptance criterion
        print(f"[decoder L{K}] traced latency measurement failed: {type(e).__name__}: {e}")
    _free(dev, kv)
    dec.deallocate()
    rope.release_prefill_tables()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="decoder-layer test helpers (host)")
    ap.add_argument("--make-goldens", action="store_true", help="build the extension goldens (module docstring)")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    if a.make_goldens:
        print(make_extension_goldens(threads=a.threads))
