# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 standalone demo on the BH Galaxy: prompts -> prefill -> greedy decode (WAVE_A_REVIEW GEN-1..7).

Runs the same runtime vLLM serves (``tt/generator.py`` ``MotifGenerator`` on a mesh opened like the plugin opens it:
``model_config.open_motif_mesh()``, FABRIC_2D_TORUS_XY, COL dispatch, trace region, L1_SMALL 32768), one user per lane
(lanes spread over the 4 DP rows), all users decoding together in the traced 32-lane decode step::

    scripts/devrun.sh -t 2400 -n demo -- python models/demos/motif3/demo/demo.py --num-layers 8 \
        --prompt "Explain mixture-of-experts routing in two sentences." --max-new-tokens 32

    # teacher-forced check against a C2 golden prompt: prefill all but the last K tokens, then decode K steps fed with
    # the prompt's own next tokens; compares the device's next-token predictions with the reference's (the golden
    # early-exit head after layer N-1 when it exists: N = 36 -> goldens/c2/final/logits_after_layer_35; otherwise the
    # reference head applied on CPU to the golden state after layer N-1, for N-1 in {0-4, 7, 8, 15, 16, 23, 24, 31, 32,
    # 35}) and with the actual next tokens
    scripts/devrun.sh -t 2400 -n demo_tf -- python models/demos/motif3/demo/demo.py --num-layers 36 \
        --teacher-force en_technical --tf-steps 32

    # build the TT weight cache for layers 0..N-1 and the globals with the production converter
    # (scripts/convert_weights.py: resumable, staged + verified, 60 GB disk floor after 1.1 x each part; ~6.65 GB per
    # MoE layer). Under devrun.sh it converts on the device; under scripts/hostrun.sh (devices hidden) on a mock mesh.
    scripts/devrun.sh -t 2400 -n convert -- python models/demos/motif3/demo/demo.py --num-layers 4 --convert

Notes: layers 36-52 of the checkpoint are not on disk yet (``--num-layers`` <= 36 until they are); the LM head always
runs (a truncated model's logits are an early-exit head, not the model's predictions). ``--cache auto`` (default) uses
the TT cache for parts a converter marked complete and reads the HF checkpoint for the rest without writing anything
(also for tensors a converted part lacks); ``--cache write`` writes missing parts, each behind the same disk guard.
Without ``scripts/convert_weights.py`` (the package copied elsewhere) ``--convert`` falls back to the in-package
``model.convert_weights`` (same parts, same disk guard, no staging / verification).

Import rule: everything from ``models.demos.motif3`` is imported inside :func:`main` (device code never at import time).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

GOLDEN_DIR = Path(os.environ.get("MOTIF3_GOLDEN_STREAM_DIR", "/home/ttuser/hchang/experiments/motif-3/goldens/c2"))
DEFAULT_PROMPTS = (
    "What is the capital of South Korea? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number.",
)


def lane_of_user(i: int, lanes_per_group: int = 8, groups: int = 4) -> int:
    """Spread users over the DP groups (the bridge's initial slot -> lane map): 0 -> 0, 1 -> 8, 2 -> 16, 3 -> 24, 4 -> 1."""
    return (i % groups) * lanes_per_group + i // groups


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-layers", type=int, default=int(os.environ.get("MOTIF3_NUM_LAYERS", "53")))
    p.add_argument("--prompt", action="append", default=None, help="user message (repeatable; chat template applied)")
    p.add_argument("--raw", action="store_true", help="--prompt strings are token-id lists / raw text (no template)")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=4096, help="largest prefill bucket (a multiple of 256)")
    p.add_argument("--teacher-force", default=None, help="C2 golden prompt name (goldens/c2/prompts.json)")
    p.add_argument("--tf-steps", type=int, default=32, help="teacher-forced decode steps")
    p.add_argument("--no-trace", action="store_true", help="eager decode (no trace capture)")
    p.add_argument("--no-warmup", action="store_true", help="skip warmup (implies --no-trace)")
    p.add_argument("--cache", default="auto", choices=("auto", "write", "off"))
    p.add_argument("--convert", action="store_true", help="only build the TT weight cache (resumable) and exit")
    p.add_argument("--kv-dtype", default=os.environ.get("MOTIF3_KV_CACHE_DTYPE", "bfp8"), choices=("bfp8", "bf16"))
    return p.parse_args(argv)


def golden_reference(name: str, num_layers: int, positions):
    """Reference next-token logits at ``positions`` of golden prompt ``name`` for a model of ``num_layers`` layers:
    ``(argmax [n], topk_ids [n, 32], topk_logits [n, 32] descending, source)`` or ``None`` when no golden exists for
    that depth."""
    from models.demos.motif3.reference import golden_stream as gs

    last = num_layers - 1
    lp, _ = gs.head_paths(GOLDEN_DIR, last)
    if lp.is_file():
        t, _ = gs.load_tensors(lp)
        return (t[f"{name}.argmax"][positions], t[f"{name}.topk_ids"][positions],
                t[f"{name}.topk_logits"][positions].float(), f"golden head after layer {last}")
    sp = gs.state_path(GOLDEN_DIR, last)
    if not sp.is_file():
        return None
    from models.demos.motif3.reference.modules import RMSNorm
    from models.demos.motif3.tt.weights import HFWeightLoader

    src = HFWeightLoader()
    x = gs.load_tensors(sp)[0][name][0][positions]  # [n, 4, 4096]
    norm = RMSNorm(4096, 1e-5).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(src.get("model.norm.weight").to(torch.bfloat16))
        logits = torch.nn.functional.linear(norm(x.mean(dim=1)), src.get("lm_head.weight").to(torch.bfloat16)).float()
    top = logits.topk(32, dim=-1)
    return logits.argmax(-1), top.indices, top.values, f"reference head on the golden state after layer {last}"


def top1_vs_golden(pred: torch.Tensor, argmax: torch.Tensor, ids_k: torch.Tensor, vals_k: torch.Tensor,
                   margin: float = 0.5) -> dict:
    """Tie-aware agreement of the predictions with the reference (a prediction agrees when it is the reference argmax
    or ties it exactly), overall and on rows whose reference top-1 - top-2 margin exceeds ``margin`` (near-ties flip
    under any change of numerics; see ``tests/test_model_truncated.py``)."""
    hit = ids_k == pred[:, None]
    at = torch.where(hit, vals_k, torch.full_like(vals_k, float("-inf"))).max(-1).values
    agree = (pred == argmax) | (at == vals_k[:, 0])
    sel = (vals_k[:, 0] - vals_k[:, 1]) > margin
    return dict(exact=float((pred == argmax).float().mean()), agree=float(agree.float().mean()),
                n_sel=int(sel.sum()), agree_sel=float(agree[sel].float().mean()) if bool(sel.any()) else float("nan"),
                in5=float(hit[:, :5].any(-1).float().mean()))


def production_converter():
    """``<project>/scripts/convert_weights.py`` when it exists, else None."""
    from models.demos.motif3.tt.model_config import PROJECT_ROOT

    p = Path(PROJECT_ROOT) / "scripts" / "convert_weights.py"
    return p if p.is_file() else None


def convert(args, num_layers: int, weights: str, src) -> int:
    """``--convert``: the TT weight cache for layers 0..N-1 + the globals. The production converter
    (``scripts/convert_weights.py``) in a child process -- ``--target device`` when the chips are visible (it checks the
    devrun.sh lock itself), ``--target mock`` when they are hidden (scripts/hostrun.sh) -- before this process opens a
    mesh; its exit code is returned (3 = disk guard). Without the script: the in-package converter on the mesh."""
    script = production_converter()
    if script is not None:
        try:
            visible = bool(os.listdir("/dev/tenstorrent"))
        except OSError:
            visible = False
        cmd = [sys.executable, str(script), "--target", "device" if visible else "mock", "--layers",
               f"0-{num_layers - 1}", "--globals", "--weights-dir", str(weights)]
        print(f"[demo] converting with the production converter: {' '.join(cmd)}", flush=True)
        return subprocess.call(cmd)
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.model import DiskGuardError, convert_weights
    from models.demos.motif3.tt.model_config import MotifTTConfig, close_motif_mesh, open_motif_mesh

    print("[demo] scripts/convert_weights.py not found: in-package converter (no staging / verification)", flush=True)
    mesh = open_motif_mesh()
    try:
        log_fabric(mesh, "demo_convert")
        cfg = MotifTTConfig.from_hf_config(mesh_device=mesh, num_layers=num_layers, kv_cache_dtype=args.kv_dtype)
        try:
            times = convert_weights(mesh, cfg, source=src, layers=range(num_layers))
        except DiskGuardError as e:
            print(f"[demo] {e}; converted before the stop: {json.dumps(getattr(e, 'converted', {}))}")
            return 3
        print(f"[demo] converted: {json.dumps(times)}")
        return 0
    finally:
        close_motif_mesh(mesh)


def main(argv=None) -> int:
    args = parse_args(argv)
    import ttnn

    from models.demos.motif3.tt import generator_api as api
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.model_config import close_motif_mesh, open_motif_mesh, resolve_weights_dir
    from models.demos.motif3.tt.weights import HFWeightLoader

    N = int(args.num_layers)
    loc = api.resolve_weights_location()
    weights = loc.path or str(resolve_weights_dir())
    try:
        src = HFWeightLoader(weights)
        missing = [l for l in range(N) if not src.layer_available(l)]
    except FileNotFoundError:
        src, missing = None, []
    if args.convert:  # the converter skips complete parts and checks the BF16 of the others itself
        return convert(args, N, weights, src)
    if missing and args.cache != "auto":
        print(f"[demo] checkpoint layers {missing[:5]}... are not on disk; use --num-layers <= {missing[0]}")
        return 2

    mesh = open_motif_mesh()
    try:
        log_fabric(mesh, "demo")
        settings = api.GeneratorSettings(
            max_batch_size=api.NUM_LANES, max_seq_len=int(args.max_model_len), num_layers=N,
            kv_cache_dtype=args.kv_dtype, weights_path=weights, block_size=api.DEFAULT_BLOCK_SIZE,
            weights_source=loc.source,
        )
        gen = MotifGenerator.create(hf_config=None, mesh_device=mesh, settings=settings, cache=args.cache)
        cfg = gen.cfg
        try:
            return run(args, gen, cfg, api, ttnn)
        finally:
            gen.close()
    finally:
        close_motif_mesh(mesh)


def run(args, gen, cfg, api, ttnn) -> int:
    N = gen.num_layers
    bs = api.DEFAULT_BLOCK_SIZE
    # ---- users ------------------------------------------------------------------------------------------------------
    users = []  # dicts: name, ids (prompt), lane
    tf = None
    if args.teacher_force:
        from models.demos.motif3.reference import golden_stream as gs

        prompts = {p.name: p for p in gs.load_prompt_set(GOLDEN_DIR / "prompts.json")}
        if args.teacher_force not in prompts:
            print(f"[demo] unknown golden prompt {args.teacher_force!r}; have {sorted(prompts)}")
            return 2
        ids = prompts[args.teacher_force].ids
        k = min(int(args.tf_steps), len(ids) - 2)
        tf = dict(name=args.teacher_force, ids=ids, S=len(ids) - k - 1, k=k)
        users.append(dict(name=args.teacher_force, ids=ids[: tf["S"]], lane=0))
    else:
        from models.demos.motif3.reference.tokenizer import encode_chat, load_tokenizer

        tok = load_tokenizer(cfg.weights_dir)
        for i, text in enumerate(args.prompt or DEFAULT_PROMPTS):
            ids = encode_chat([{"role": "user", "content": text}], tok) if not args.raw else tok.encode(text)
            users.append(dict(name=f"user{i}", ids=ids, lane=lane_of_user(i), text=text))
        if len(users) > api.NUM_LANES:
            print(f"[demo] at most {api.NUM_LANES} prompts")
            return 2
    steps = tf["k"] if tf else int(args.max_new_tokens)
    for u in users:
        if len(u["ids"]) + steps > cfg.max_model_len:
            print(f"[demo] {u['name']}: {len(u['ids'])} + {steps} tokens exceed --max-model-len {cfg.max_model_len}")
            return 2

    # ---- KV pool: every user's blocks + the null block -----------------------------------------------------------
    need = [math.ceil((len(u["ids"]) + steps) / bs) for u in users]
    num_blocks = 1 + sum(need) + 1
    pool = gen.allocate_kv_cache(num_blocks=num_blocks, block_size=bs, num_layers=N)
    W = min(math.ceil(cfg.max_model_len / bs), num_blocks)
    nxt = 1
    for u, n in zip(users, need):
        u["blocks"] = list(range(nxt, nxt + n))
        nxt += n

    # ---- warmup (all buckets before the capture) --------------------------------------------------------------------
    trace = not (args.no_trace or args.no_warmup)
    t0 = time.time()
    if not args.no_warmup:
        gen.warmup_prefill(kv_cache=pool, enable_trace=False)
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=W)
        if trace:
            gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
        print(f"[demo] warmup ({len(cfg.prefill_buckets)} buckets{', decode trace' if trace else ''}) "
              f"{time.time() - t0:.1f} s")

    # ---- prefill ---------------------------------------------------------------------------------------------------
    for u in users:
        S = len(u["ids"])
        pt = torch.zeros(W, dtype=torch.int32)
        own = math.ceil(S / bs)
        pt[:own] = torch.tensor(u["blocks"][:own], dtype=torch.int32)
        t1 = time.time()
        logits = gen.prefill_forward(api.PrefillRequest(lane=u["lane"], tokens=torch.tensor(u["ids"], dtype=torch.int32),
                                                        page_table=pt), kv_cache=pool)
        u["ttft"] = time.time() - t1
        u["out"] = [int(logits.float().argmax())]
        u["pos"] = S
        print(f"[demo] {u['name']} (lane {u['lane']}): prefill {S} tokens (bucket {cfg.prefill_bucket(S)}) "
              f"{u['ttft'] * 1e3:.0f} ms")

    # ---- decode (all users in one 32-lane step) ------------------------------------------------------------------
    eos = set(cfg.eos_token_ids)
    tf_pred = []
    step_ms = []
    for t in range(steps):
        tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
        pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
        table = torch.zeros(api.NUM_LANES, W, dtype=torch.int32)
        live = [u for u in users if not u.get("done")]
        if not live:
            break
        for u in live:
            nb = u["pos"] // bs + 1
            tokens[u["lane"]] = tf["ids"][u["pos"]] if tf else u["out"][-1]
            pos[u["lane"]] = u["pos"]
            table[u["lane"], :nb] = torch.tensor(u["blocks"][:nb], dtype=torch.int32)
        t1 = time.perf_counter()
        logits = gen.decode_forward(api.DecodeBatch(tokens=tokens, positions=pos, page_table=table), kv_cache=pool,
                                    enable_trace=trace)
        step_ms.append((time.perf_counter() - t1) * 1e3)
        for u in live:
            nxt_tok = int(logits[u["lane"]].float().argmax())
            u["out"].append(nxt_tok)
            if tf:
                tf_pred.append(nxt_tok)
            u["pos"] += 1
            if not tf and nxt_tok in eos:
                u["done"] = True

    # ---- report ----------------------------------------------------------------------------------------------------
    med = statistics.median(step_ms) if step_ms else float("nan")
    print(f"[demo] {N} layers: decode {len(step_ms)} steps, median {med:.1f} ms/step ({1e3 / med:.1f} tok/s/user, "
          f"{len(users)} users; {'traced' if trace else 'eager'})")
    if tf:
        S, k, ids = tf["S"], tf["k"], tf["ids"]
        pred = [users[0]["out"][0]] + tf_pred  # predictions for positions S .. S + k
        positions = list(range(S - 1, S + k))  # position p's logits predict token p + 1
        targets = torch.tensor(ids[S: S + k + 1])
        dev = torch.tensor(pred[: len(targets)])
        acc = float((dev == targets).float().mean())
        print(f"[demo] teacher-forced {tf['name']}: {len(dev)} predictions, device top-1 == next token {acc:.3f}")
        ref = golden_reference(tf["name"], N, positions[: len(dev)])
        if ref is not None:
            am, ids_k, vals_k, srcname = ref
            m = top1_vs_golden(dev, am, ids_k, vals_k)
            racc = float((am == targets).float().mean())
            print(f"[demo] vs {srcname}: device argmax == reference argmax {m['exact']:.3f} (tie-aware {m['agree']:.3f}; "
                  f"{m['agree_sel']:.3f} on the {m['n_sel']} rows with a reference margin > 0.5); device argmax in the "
                  f"reference top-5 {m['in5']:.3f}; reference top-1 == next token {racc:.3f}")
        else:
            print(f"[demo] no golden for a {N}-layer model (states after layers {{0-4,7,8,15,16,23,24,31,32,35}})")
    else:
        from models.demos.motif3.reference.tokenizer import load_tokenizer

        tok = load_tokenizer(cfg.weights_dir)
        for u in users:
            text = tok.decode(u["out"], skip_special_tokens=False)
            print(f"[demo] {u['name']}: {u.get('text', '')!r}\n        -> {text!r}")
        if N < cfg.num_hidden_layers:
            print(f"[demo] NOTE: a {N}-layer truncated model: the LM head is an early-exit head (text is not meaningful)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
