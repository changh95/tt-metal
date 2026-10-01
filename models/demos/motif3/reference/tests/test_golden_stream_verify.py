# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent verification of ``golden_stream.py`` and of the production streaming golden.

Complements ``test_golden_stream.py`` (fresh runs vs the reference forward on a tiny model and on real layers 0-3).
Here the checks are (a) the resume path on real weights at more depth and (b) the files the production run wrote:
``MOTIF3_GOLDEN_STREAM_DIR`` (default ``/home/ttuser/hchang/experiments/motif-3/goldens/c2``). Every test skips by
itself when the golden, or the shards it needs, are absent, e.g. once the early shards are deleted. The tests do not
set the torch thread count. The production run used 32 threads, and the checks below are bit-exact at 16 as well.

1. Layer schedule of all 53 layers: reference ``DecoderLayer`` == official HF ``MotifDecoderLayer`` (both on the meta
   device) == an independent formula (global iff ``L % 4 == 0``, dense iff ``L < 2``) == the production manifest.
2. Resume == uninterrupted run on real weights: layers 0-5 in one call vs 0-2 and then ``resume`` 3-5 (plus the
   early-exit head after layer 5). The two runs are compared file by file, bit for bit. Both also equal
   ``MotifForCausalLM.forward`` of ``load_reference_model(range(6))`` and the production states.
3. Resume from the production golden at depth with the early shards absent. A symlinked checkpoint holds only the
   shards of layers 32-35 and of the head. The production ``after_layer_31`` is the resume checkpoint and
   ``layers="auto"`` is used. Layers 32-35 and the head after 35 must be bit-identical to the uninterrupted run.
4. Production bookkeeping: manifest, per-file metadata, the frozen prompt set (== ``prompts/rendered.json``), the
   recorded per-layer schedule and stats, file sizes, the disk budget, and ``resume == after_layer_<last>``.
5. Prompt rendering: an independent re-implementation of ``chat_template.jinja`` (text, then token ids and roles via
   per-piece tokenization) and ``PreTrainedTokenizerFast.from_pretrained`` (``tokenizer_config.json``) reproduce
   ``rendered.json``. Each prompt has exactly one BOS, and ``<|endoftext|>`` is only the last token of complete
   conversations.
6. Every saved transition ``after_{L-1} -> after_L`` is recomputed with a reference ``DecoderLayer`` built here with
   materialized (memory-mapped) experts, not golden_stream's lazy-expert loader.
7. The official HF ``MotifDecoderLayer`` (``hf_reference.py`` CPU flash shim) on stored states reproduces the stored
   next states: embed -> layer 0, layers 5-7 (the onset of python_code's outlier token), layer 8 (global/moe) and
   layers 33-35 (swa/moe). ``MOTIF3_VERIFY_FULL_CHAIN=1`` (~2.5 min) recomputes the whole chain: ids -> embedding ->
   HF layers 0..last -> head, compared with every saved file.
8. Every stored head (early-exit or final) recomputed from the stored state with HF's ``MotifRMSNorm`` and the
   checkpoint's ``lm_head``, plus the manifest's summary metrics.
9. Sanity of the stored states: finite, smooth growth of the typical token scale, causality (shared prompt prefixes
   give the same states up to bf16 noise), bf16 tracks the fp32 sanity run. Outlier tokens are reported.

Run (from the tt-metal root; add ``-rA -s`` for the printed numbers)::

    python_env/bin/python -m pytest -p no:cacheprovider --noconftest -o addopts="" --import-mode=importlib \
        models/demos/motif3/reference/tests/test_golden_stream_verify.py
"""

from __future__ import annotations

import copy
import gc
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.motif3.reference import MotifCheckpoint, load_reference_model
from models.demos.motif3.reference.config import MotifArgs
from models.demos.motif3.reference.golden import TensorRecorder, pcc
from models.demos.motif3.reference.golden_stream import (
    DEFAULT_MESSAGES,
    DEFAULT_OUT_DIR,
    DEFAULT_PROMPT_SET,
    DEFAULT_SAVE_LAYERS,
    FORMAT,
    TOPK,
    head_paths,
    load_prompt_set,
    load_tensors,
    prompt_set_sha256,
    run_stream,
    state_path,
)
from models.demos.motif3.reference.modules import DecoderLayer
from models.demos.motif3.reference.weights import DEFAULT_WEIGHTS_DIR

from .hf_reference import HF_META_DIR, hf_config_from_json, hf_cpu_flash_attention, load_hf_modules

pytestmark = pytest.mark.timeout(3600)

GOLDEN_DIR = Path(DEFAULT_OUT_DIR)
FP32_SANITY_DIR = GOLDEN_DIR / "fp32_sanity_L0-7"
N_LAYERS = 53
D = 4096
E = 4
BOS_ID, EOS_ID = 1, 0
DISK_BUDGET = 5e9
RESUME_PROMPTS = ("chat_default", "en_technical", "python_code")  # 145 / 893 / 552 tokens


def _quiet(msg):
    pass


def _identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise-equal values, dtype and shape (NaN == NaN)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        return torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(0.0), b.nan_to_num(0.0))
    return torch.equal(a, b)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().nan_to_num(0.0).max())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_mismatches(path_a: Path, path_b: Path) -> list:
    """Keys of two golden-stream files that are not bit-identical (plus a metadata marker)."""
    ta, ma = load_tensors(path_a)
    tb, mb = load_tensors(path_b)
    bad = sorted(set(ta) ^ set(tb))
    bad += [k for k in sorted(set(ta) & set(tb)) if not _identical(ta[k], tb[k])]
    if ma != mb:
        bad.append("<metadata>")
    return bad


def _run_files(out: Path) -> list:
    return sorted(
        str(p.relative_to(out)) for sub in ("states", "resume", "final") for p in (out / sub).glob("*.safetensors")
    )


def _expected_kind(L: int) -> dict:
    """Independent layer schedule (Motif-3 config: sliding_window_period 4, n_dense_first_layers 2, window 128 + 1)."""
    swa = L % 4 != 0
    mscale = 0.1 * 1.0 * torch.log(torch.tensor(64.0, dtype=torch.float64)).item() + 1.0
    return dict(
        is_swa=swa,
        window=129 if swa else None,
        softmax_scale=192**-0.5 * (1.0 if swa else mscale * mscale),
        uses_yarn=not swa,
        is_moe=L >= 2,
        kind=("swa" if swa else "global") + "/" + ("moe" if L >= 2 else "dense"),
    )


def _checkpoint_or_skip(layers=(), extra=()) -> MotifCheckpoint:
    """The local checkpoint if every tensor of ``layers`` and ``extra`` is complete on disk, else skip."""
    try:
        ck = MotifCheckpoint(DEFAULT_WEIGHTS_DIR)
    except FileNotFoundError as e:
        pytest.skip(f"checkpoint not available: {e}")
    names = list(extra) + [n for L in layers for n in ck.layer_names(L)]
    missing = [n for n in names if not ck.is_local(n)]
    if missing:
        pytest.skip(f"shards not local: {missing[:2]}...")
    return ck


def _ref_layer(args: MotifArgs, L: int) -> DecoderLayer:
    """Reference ``DecoderLayer`` with materialized (memory-mapped) experts, through its own checkpoint handles (the
    shards are unmapped when the layer is dropped). Deliberately not golden_stream's lazy-expert loader."""
    ck = MotifCheckpoint(DEFAULT_WEIGHTS_DIR)
    with torch.device("meta"):
        layer = DecoderLayer(args, L, materialize_experts=True)
    layer.load_state_dict(ck.layer_state_dict(L, torch.bfloat16, strip_prefix=True), strict=True, assign=True)
    layer.requires_grad_(False)
    return layer


def _hf_layer(mm, cfg, L: int):
    """Official HF ``MotifDecoderLayer`` holding the raw checkpoint tensors (HF names, no reference name mapping)."""
    ck = MotifCheckpoint(DEFAULT_WEIGHTS_DIR)
    p = f"model.layers.{L}."
    sd = {n[len(p) :]: ck.get(n) for n in ck.layer_names(L)}
    with torch.device("meta"):
        layer = mm.MotifDecoderLayer(cfg, L)
    layer.load_state_dict(sd, strict=True, assign=True)
    layer.requires_grad_(False)
    return layer.eval()


def _hf_apply(layer, rotary, x: torch.Tensor) -> torch.Tensor:
    """One HF decoder layer as ``MotifModel.forward`` calls it (B=1, positions 0..S-1, no cache, model-level YaRN
    cos/sin; SWA layers use their own plain RoPE)."""
    S = x.shape[1]
    pos = torch.arange(S)[None]
    return layer(
        x,
        attention_mask=None,
        position_ids=pos,
        past_key_value=None,
        use_cache=False,
        cache_position=pos[0],
        position_embeddings=rotary(x[:, :, 0], pos),
    )[0]


def _positions(x: torch.Tensor) -> torch.Tensor:
    B, S = x.shape[:2]
    return torch.arange(S)[None, :].expand(B, S)


# =================================================================================================
# production golden fixture
# =================================================================================================
@pytest.fixture(scope="module")
def golden():
    path = GOLDEN_DIR / "manifest.json"
    if not path.exists():
        pytest.skip(f"no streaming golden at {GOLDEN_DIR}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    prompts = load_prompt_set(GOLDEN_DIR / "prompts.json")
    return SimpleNamespace(dir=GOLDEN_DIR, manifest=manifest, prompts=prompts, by_name={p.name: p for p in prompts})


def _stored_state(golden, L: int):
    """``{name: [1, S, 4, D]}`` after layer ``L`` (``-1``: the layer-0 input, i.e. 4 copies of the embedding);
    ``None`` if not saved. The resume checkpoint stands in for the last processed layer."""
    if L < 0:
        emb = load_tensors(golden.dir / "states" / "embed.safetensors")[0]
        return {n: e.unsqueeze(2).expand(-1, -1, E, -1).contiguous() for n, e in emb.items()}
    path = state_path(golden.dir, L)
    if path.exists():
        return load_tensors(path)[0]
    resume = golden.dir / "resume" / "state.safetensors"
    if resume.exists():
        tensors, meta = load_tensors(resume)
        if meta.get("last_layer") == L:
            return tensors
    return None


# =================================================================================================
# 1. layer schedule
# =================================================================================================
def test_layer_schedule_reference_vs_hf_all_layers():
    if not (DEFAULT_WEIGHTS_DIR / "config.json").exists():
        pytest.skip("config.json not available")
    args = MotifArgs.from_hf_config(DEFAULT_WEIGHTS_DIR)
    assert args.num_hidden_layers == N_LAYERS
    cfg_mod, mm = load_hf_modules()
    cfg = hf_config_from_json(DEFAULT_WEIGHTS_DIR, cfg_mod)
    kinds = []
    for L in range(N_LAYERS):
        exp = _expected_kind(L)
        with torch.device("meta"):
            ref = DecoderLayer(args, L)
            hf = mm.MotifDecoderLayer(cfg, L)
        a, h = ref.self_attn, hf.self_attn
        assert a.is_swa == h.is_swa_layer == exp["is_swa"], L
        assert a.window == h.sliding_window == exp["window"], L
        assert abs(a.scale - exp["softmax_scale"]) < 1e-12 and abs(h.scaling - exp["softmax_scale"]) < 1e-12, L
        assert a.uses_yarn == exp["uses_yarn"] == (h.swa_rotary_emb is None), L
        assert a.rope_theta == 10000.0
        assert ref.is_moe == hf.moe_enabled == exp["is_moe"], L
        assert args.layer_kind(L) == exp["kind"]
        kinds.append(exp["kind"])
    assert [L for L in range(N_LAYERS) if kinds[L].startswith("global")] == list(range(0, N_LAYERS, 4))  # 14 global
    assert [L for L in range(N_LAYERS) if kinds[L].endswith("dense")] == [0, 1]
    # the RoPE tables of both layer kinds
    from models.demos.motif3.reference.rope import inv_freq_for_layer

    swa_cfg = copy.copy(cfg)
    swa_cfg.rope_scaling = None
    swa_cfg.rope_theta = cfg.swa_rope_theta
    hf_swa = mm.MotifRotaryEmbedding(swa_cfg, rope_head_dim=cfg.qk_rope_head_dim).inv_freq
    hf_global = mm.MotifRotaryEmbedding(cfg, rope_head_dim=cfg.qk_rope_head_dim).inv_freq
    assert torch.equal(inv_freq_for_layer(args, 1).float(), hf_swa.float())
    assert torch.equal(inv_freq_for_layer(args, 4).float(), hf_global.float())
    assert not torch.equal(hf_swa, hf_global)
    # the HF sources the oracle imports are the checkpoint's own modeling file
    ckpt_src = DEFAULT_WEIGHTS_DIR / "modeling_motif.py"
    if ckpt_src.exists():
        assert _sha256(ckpt_src) == _sha256(HF_META_DIR / "modeling_motif.py")


def test_production_manifest_layer_schedule(golden):
    for e in golden.manifest["layer_log"]:
        exp = _expected_kind(e["layer_idx"])
        assert e["kind"] == exp["kind"] and e["is_swa"] == exp["is_swa"] and e["window"] == exp["window"]
        assert e["uses_yarn"] == exp["uses_yarn"] and e["is_moe"] == exp["is_moe"]
        assert abs(e["softmax_scale"] - exp["softmax_scale"]) < 1e-12


# =================================================================================================
# 2. resume == uninterrupted on real weights (layers 0-5)
# =================================================================================================
def test_real_resume_after_2_equals_straight_0_5(tmp_path):
    ck = _checkpoint_or_skip(range(6), extra=("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"))
    prompts = load_prompt_set(DEFAULT_PROMPT_SET, RESUME_PROMPTS)
    straight, resumed = tmp_path / "straight", tmp_path / "resumed"
    kw = dict(ckpt_dir=ck.dir, save_layers=None, log=_quiet)
    try:
        m_s = run_stream(straight, "0-5", prompts=prompts, prompt_set_path=DEFAULT_PROMPT_SET, final_head=True, **kw)
        m_a = run_stream(resumed, "0-2", prompts=prompts, prompt_set_path=DEFAULT_PROMPT_SET, **kw)
        assert m_a["last_layer"] == 2 and not m_a["heads"]
        assert load_tensors(resumed / "resume" / "state.safetensors")[1]["last_layer"] == 2
        m_r = run_stream(resumed, "3-5", resume=True, final_head=True, **kw)

        # the two runs: same files, bit for bit (tensors and metadata), same bookkeeping
        files = _run_files(straight)
        assert files == _run_files(resumed)
        assert len(files) == 1 + 6 + 1 + 2  # embed, after_layer_00..05, resume, logits + hidden after 5
        bad = {rel: _file_mismatches(straight / rel, resumed / rel) for rel in files}
        assert not any(bad.values()), bad
        assert (straight / "prompts.json").read_bytes() == (resumed / "prompts.json").read_bytes()
        for key in ("format", "mode", "checkpoint", "prompt_set", "layers_done", "last_layer", "saved_layers", "files"):
            assert m_s[key] == m_r[key], key
        assert m_s["heads"]["5"]["summary"] == m_r["heads"]["5"]["summary"] and m_r["heads"]["5"]["early_exit"]
        assert [e["stats"] for e in m_s["layer_log"]] == [e["stats"] for e in m_r["layer_log"]]
        assert len(m_r["runs"]) == 2 and m_r["runs"][1]["resume"] and m_r["runs"][1]["layers"] == [3, 5]

        # both equal the reference's ordinary full forward of the 6-layer prefix model
        model = load_reference_model(ck.dir, layer_ids=range(6), dtype=torch.bfloat16, lazy_experts=True, checkpoint=ck)
        lt = load_tensors(head_paths(straight, 5)[0])[0]
        ht = load_tensors(head_paths(straight, 5)[1])[0]
        prod = {}  # production states of the same prompts (same token ids), where saved
        if (GOLDEN_DIR / "prompts.json").exists():
            prod_ids = {q.name: q.ids for q in load_prompt_set(GOLDEN_DIR / "prompts.json")}
            same = {p.name for p in prompts if prod_ids.get(p.name) == p.ids}
            for L in range(6):
                if state_path(GOLDEN_DIR, L).exists():
                    prod[L] = {n: t for n, t in load_tensors(state_path(GOLDEN_DIR, L))[0].items() if n in same}
        for p in prompts:
            rec = TensorRecorder(lambda n: n.endswith(".x_out") or n in ("final.stream_mean", "final.norm", "logits"))
            with torch.no_grad():
                model(torch.tensor([p.ids]), tap=rec)
            ref = rec.tensors
            for L in range(6):
                got = load_tensors(state_path(straight, L))[0][p.name]
                assert torch.equal(got, ref[f"layers.{L}.x_out"]), (p.name, L, _max_abs(got, ref[f"layers.{L}.x_out"]))
                if prod.get(L) is not None and p.name in prod[L]:
                    assert torch.equal(got, prod[L][p.name]), ("production", p.name, L)
            logits = ref["logits"][0]
            assert torch.equal(ht[f"{p.name}.final_hidden"], ref["final.norm"].float())
            assert torch.equal(ht[f"{p.name}.stream_mean"], ref["final.stream_mean"].float())
            top_v, top_i = torch.topk(logits, TOPK, dim=-1)
            assert torch.equal(lt[f"{p.name}.topk_ids"], top_i) and torch.equal(lt[f"{p.name}.topk_logits"], top_v)
            assert torch.equal(lt[f"{p.name}.logsumexp"], torch.logsumexp(logits, -1))
        print(
            f"\nlayers 0-5 straight == 0-2 + resume 3-5 == prefix forward ({len(files)} files, {len(prompts)} prompts)"
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


# =================================================================================================
# 3. resume from the production golden at depth, early shards absent
# =================================================================================================
RESUME_K, RESUME_LAYERS = 31, (32, 33, 34, 35)


def test_production_resume_after_31_without_early_shards(golden, tmp_path):
    m = golden.manifest
    if m["last_layer"] is None or m["last_layer"] < RESUME_LAYERS[-1]:
        pytest.skip("production golden does not reach layer 35")
    for L in (RESUME_K, 32, 35):
        if not state_path(golden.dir, L).exists():
            pytest.skip(f"production state after layer {L} not saved")
    head_l, head_h = head_paths(golden.dir, 35)
    if not head_l.exists():
        pytest.skip("no head after layer 35 in the production golden")
    head_names = ("model.norm.weight", "lm_head.weight")
    ck = _checkpoint_or_skip(RESUME_LAYERS, extra=head_names)

    # a checkpoint directory holding only what layers 32-35 and the head need (symlinks; nothing is copied)
    part = tmp_path / "ckpt"
    part.mkdir()
    shards = {ck.weight_map[n] for L in RESUME_LAYERS for n in ck.layer_names(L)} | {
        ck.weight_map[n] for n in head_names
    }
    assert ck.weight_map["model.embed_tokens.weight"] not in shards
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        if (ck.dir / name).exists():
            (part / name).symlink_to(ck.dir / name)
    for fn in shards:
        (part / fn).symlink_to(ck.dir / fn)
    (part / ".download_state.json").write_text(json.dumps({"complete_layers": list(RESUME_LAYERS)}))

    # the output directory as the production run left it right after layer 31
    out = tmp_path / "out"
    (out / "resume").mkdir(parents=True)
    shutil.copyfile(state_path(golden.dir, RESUME_K), out / "resume" / "state.safetensors")
    shutil.copyfile(golden.dir / "prompts.json", out / "prompts.json")
    man = copy.deepcopy(m)
    man.update(
        last_layer=RESUME_K,
        layers_done=list(range(RESUME_K + 1)),
        layer_log=[e for e in m["layer_log"] if e["layer_idx"] <= RESUME_K],
        heads={},
        runs=m["runs"][:1],
        saved_layers=[],
        files={},
    )
    (out / "manifest.json").write_text(json.dumps(man))
    try:
        res = run_stream(out, "auto", resume=True, ckpt_dir=part, save_layers=None, final_head=True, log=_quiet)
        assert res["last_layer"] == 35 and res["layers_done"] == list(range(36)) and list(res["heads"]) == ["35"]
        for L in (32, 35):
            assert not _file_mismatches(state_path(out, L), state_path(golden.dir, L)), L
        assert not _file_mismatches(out / "resume" / "state.safetensors", state_path(golden.dir, 35))
        lp, hp = head_paths(out, 35)
        assert not _file_mismatches(lp, head_l) and not _file_mismatches(hp, head_h)
        assert res["heads"]["35"]["summary"] == m["heads"]["35"]["summary"]
        prod_stats = {e["layer_idx"]: e["stats"] for e in m["layer_log"]}
        for e in res["layer_log"]:  # the stats are fp32 full reductions: last-digit rounding may follow the threads
            for n, s in e["stats"].items():
                ref = prod_stats[e["layer_idx"]][n]
                assert s["finite"] == ref["finite"] and s["absmax"] == ref["absmax"], (e["layer_idx"], n)
                assert abs(s["rms"] - ref["rms"]) <= 2e-6 + 1e-5 * ref["rms"], (e["layer_idx"], n)
        print("\nresume from production after_layer_31 with only layers 32-35 + head shards: bit-identical to the run")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


# =================================================================================================
# 4. production bookkeeping
# =================================================================================================
def _stats(x: torch.Tensor) -> dict:
    xf = x.double()
    return dict(rms=float(xf.pow(2).mean().sqrt()), absmax=float(xf.abs().max()), finite=bool(torch.isfinite(xf).all()))


def test_production_bookkeeping(golden):
    m, d = golden.manifest, golden.dir
    assert m["format"] == FORMAT
    mode = {k: m["mode"][k] for k in ("dtype", "attn_mode", "q_path_fp32", "mhc_mix_fp32")}
    assert mode == dict(dtype="bf16", attn_mode="expanded", q_path_fp32=True, mhc_mix_fp32=False)
    assert m["checkpoint"]["num_hidden_layers"] == N_LAYERS
    cfg_path = DEFAULT_WEIGHTS_DIR / "config.json"
    if cfg_path.exists():
        assert m["checkpoint"]["config_sha256"] == _sha256(cfg_path), "config.json changed since the run"

    # the frozen prompt set is the current prompts/rendered.json
    sha = prompt_set_sha256(golden.prompts)
    frozen = json.loads((d / "prompts.json").read_text(encoding="utf-8"))
    assert m["prompt_set"]["sha256"] == frozen["sha256"] == sha
    assert m["prompt_set"]["prompts"] == [dict(name=p.name, n_tokens=len(p.ids)) for p in golden.prompts]
    assert m["prompt_set"]["n_tokens_total"] == sum(len(p.ids) for p in golden.prompts)
    rendered = json.loads(DEFAULT_PROMPT_SET.read_text(encoding="utf-8"))
    assert rendered["sha256"] == sha, "prompts/rendered.json changed since the golden was computed"
    for a, b in zip(frozen["prompts"], rendered["prompts"]):
        assert (a["name"], a["ids"], a["roles"], a["text"]) == (b["name"], b["ids"], b["roles"], b["text"])

    # layers done / saved
    last = m["last_layer"]
    assert last is not None and m["layers_done"] == list(range(last + 1))
    assert [e["layer_idx"] for e in m["layer_log"]] == list(range(last + 1))
    on_disk = sorted(int(p.stem.rsplit("_", 1)[1]) for p in (d / "states").glob("after_layer_*.safetensors"))
    assert m["saved_layers"] == on_disk
    for r in m["runs"]:
        if r["save_layers"] != "all" and r["layers"]:
            want = set(r["save_layers"]) & set(range(r["layers"][0], r["layers"][1] + 1))
            assert want <= set(on_disk), sorted(want - set(on_disk))
    assert set(on_disk) >= set(DEFAULT_SAVE_LAYERS) & set(range(last + 1))

    # sizes and disk budget
    for rel, size in m["files"].items():
        assert (d / rel).stat().st_size == size, rel
    assert m["total_bytes"] == sum(m["files"].values())
    assert set(m["files"]) == set(_run_files(d)) | {"prompts.json"}
    du = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    print(f"\n{d}: {du / 1e9:.2f} GB on disk (incl. sub-runs), manifest total {m['total_bytes'] / 1e9:.2f} GB")
    assert du < DISK_BUDGET
    assert not list(d.rglob("*.tmp")), "leftover temp files"

    # per-file metadata, keys, shapes, dtypes and the recorded stats
    seq = {p.name: len(p.ids) for p in golden.prompts}
    stats = {e["layer_idx"]: e["stats"] for e in m["layer_log"]}
    for L in on_disk:
        tensors, meta = load_tensors(state_path(d, L))
        assert meta == dict(
            format=FORMAT,
            kind="x_out [1, S, E=4, D]",
            layer=L,
            last_layer=L,
            mode=m["mode"],
            prompt_sha256=sha,
            config_sha256=m["checkpoint"]["config_sha256"],
        )
        assert set(tensors) == set(seq)
        for n, x in tensors.items():
            assert x.dtype == torch.bfloat16 and x.shape == (1, seq[n], E, D), (L, n)
            s, rec = _stats(x), stats[L][n]
            assert rec["finite"] == s["finite"] and rec["absmax"] == round(s["absmax"], 6), (L, n)
            assert abs(rec["rms"] - s["rms"]) <= 2e-6 + 1e-5 * s["rms"], (L, n, rec["rms"], s["rms"])
    resume_t, resume_meta = load_tensors(d / "resume" / "state.safetensors")
    assert resume_meta["last_layer"] == last and resume_meta["prompt_sha256"] == sha
    if last in on_disk:
        last_t = load_tensors(state_path(d, last))[0]
        assert set(resume_t) == set(last_t) and all(torch.equal(resume_t[n], last_t[n]) for n in last_t)
    emb, emb_meta = load_tensors(d / "states" / "embed.safetensors")
    assert emb_meta["format"] == FORMAT and emb_meta["mode"] == m["mode"] and set(emb) == set(seq)
    assert all(emb[n].dtype == torch.bfloat16 and emb[n].shape == (1, seq[n], D) for n in seq)

    # heads
    keys = (
        "topk_ids topk_logits argmax argmax_ties logsumexp target_ids target_logit target_logprob target_rank "
        "target_in_assistant"
    ).split()
    for k, h in m["heads"].items():
        lp, hp = head_paths(d, int(k))
        assert h["logits"] == str(lp.relative_to(d)) and h["hidden"] == str(hp.relative_to(d))
        lt, lmeta = load_tensors(lp)
        ht, hmeta = load_tensors(hp)
        assert lmeta == hmeta and lmeta["after_layer"] == int(k) and lmeta["topk"] == TOPK
        assert lmeta["early_exit"] == h["early_exit"] == (int(k) != N_LAYERS - 1) and lmeta["prompt_sha256"] == sha
        assert set(lt) == {f"{n}.{q}" for n in seq for q in keys}
        assert set(ht) == {f"{n}.{q}" for n in seq for q in ("final_hidden", "stream_mean")}
        for n, S in seq.items():
            assert lt[f"{n}.topk_ids"].shape == (S, TOPK) and lt[f"{n}.topk_logits"].dtype == torch.float32
            assert ht[f"{n}.final_hidden"].shape == (1, S, D) and ht[f"{n}.final_hidden"].dtype == torch.float32


# =================================================================================================
# 5. prompt rendering, independently
# =================================================================================================
_ROLE = {"system": 1, "user": 2, "assistant": 3}
_THINK = 4
_TAG = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
_SPECIAL_PIECES = (
    "<|beginoftext|>",
    "<|endoftext|>",
    "<|startofturn|>",
    "<|endofturn|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<think>",
    "</think>",
)


def _chat_pieces(messages, add_generation_prompt: bool):
    """``[(text, role)]`` of ``chat_template.jinja`` for system/user/assistant turns with string content and an
    optional ``reasoning_content`` (no tools / references / inline <think>), written from the template by hand:
    BOS, turns ``<|startofturn|><|role|>content<|endofturn|>``, the assistant's ``<think>reasoning</think>`` only in
    the last assistant turn, assistant content stripped, then the generation prompt
    ``<|startofturn|><|assistant|><think>`` or ``<|endoftext|>``. Roles follow ``golden_stream.ROLE_NAMES``."""
    pieces = [("<|beginoftext|>", 0)]
    last_asst = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    for i, msg in enumerate(messages):
        role = _ROLE[msg["role"]]
        pieces += [("<|startofturn|>", 0), (_TAG[msg["role"]], 0)]
        if msg["role"] == "assistant":
            reasoning = msg.get("reasoning_content") or ""
            if reasoning and i == last_asst:
                pieces += [("<think>", _THINK), (reasoning.strip(), _THINK), ("</think>", _THINK)]
            if msg["content"].strip():
                pieces.append((msg["content"].strip(), role))
        else:
            pieces.append((msg["content"], role))
        pieces.append(("<|endofturn|>", role))
    if add_generation_prompt:
        pieces += [("<|startofturn|>", 0), ("<|assistant|>", 0), ("<think>", _THINK)]
    else:
        pieces.append(("<|endoftext|>", 0))
    return pieces


@pytest.fixture(scope="module")
def tokenizer():
    if not (DEFAULT_WEIGHTS_DIR / "tokenizer.json").exists():
        pytest.skip("tokenizer files not available")
    from models.demos.motif3.reference.tokenizer import load_tokenizer

    return load_tokenizer()


def test_prompt_rendering_independent(tokenizer):
    from transformers import PreTrainedTokenizerFast

    src = json.loads(DEFAULT_MESSAGES.read_text(encoding="utf-8"))["prompts"]
    rendered = json.loads(DEFAULT_PROMPT_SET.read_text(encoding="utf-8"))
    doc = rendered["prompts"]
    assert [p["name"] for p in src] == [p["name"] for p in doc]
    # rendered with the checkpoint's current tokenizer and chat template
    for key, fn in (("tokenizer_json_sha256", "tokenizer.json"), ("chat_template_sha256", "chat_template.jinja")):
        assert rendered["tokenizer"][key] == _sha256(DEFAULT_WEIGHTS_DIR / fn), fn
    tok_cfg = PreTrainedTokenizerFast.from_pretrained(str(DEFAULT_WEIGHTS_DIR))  # tokenizer_config.json + template
    assert tok_cfg.bos_token_id == BOS_ID and tok_cfg.eos_token_id == EOS_ID
    special = {s: tokenizer.convert_tokens_to_ids(s) for s in _SPECIAL_PIECES}
    assert len(set(special.values())) == len(special)
    for sp, rp in zip(src, doc):
        gen = bool(sp.get("add_generation_prompt", False))
        assert rp["add_generation_prompt"] == gen and rp["messages"] == sp["messages"]
        for msg in sp["messages"]:
            assert set(msg) <= {"role", "content", "reasoning_content"} and isinstance(msg["content"], str)
            assert "<think>" not in msg["content"] and "</think>" not in msg["content"]
        pieces = _chat_pieces(sp["messages"], gen)
        text = "".join(t for t, _ in pieces)
        assert text == rp["text"], sp["name"]
        ids, roles = [], []
        for t, r in pieces:
            piece_ids = [special[t]] if t in special else tokenizer(t, add_special_tokens=False)["input_ids"]
            ids += piece_ids
            roles += [r] * len(piece_ids)
        assert ids == rp["ids"], sp["name"]
        assert roles == rp["roles"], sp["name"]
        # the tokenizer as tokenizer_config.json defines it: same template text, no automatic BOS
        assert tok_cfg.apply_chat_template(sp["messages"], add_generation_prompt=gen, tokenize=False) == text
        assert tok_cfg(text)["input_ids"] == ids
        assert ids.count(BOS_ID) == 1 and ids[0] == BOS_ID
        if gen:
            assert EOS_ID not in ids and ids[-3:] == [
                special["<|startofturn|>"],
                special["<|assistant|>"],
                special["<think>"],
            ]
        else:
            assert ids.index(EOS_ID) == len(ids) - 1 and ids[-2] == special["<|endofturn|>"]
        # every message's text is in its role's region
        for msg in sp["messages"]:
            body = msg["content"].strip() if msg["role"] == "assistant" else msg["content"]
            region = tokenizer.decode([t for t, r in zip(ids, roles) if r == _ROLE[msg["role"]]])
            assert body in region, (sp["name"], msg["role"])


# =================================================================================================
# 6. saved transitions vs reference layers (materialized experts)
# =================================================================================================
REF_PAIRS = [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (7, 8), (15, 16), (23, 24), (31, 32), (51, 52)]


@pytest.mark.parametrize("pair", REF_PAIRS, ids=[f"{a}to{b}" for a, b in REF_PAIRS])
def test_production_transition_matches_reference_layer(golden, pair):
    a, b = pair
    x_in, x_out = _stored_state(golden, a), _stored_state(golden, b)
    if x_in is None or x_out is None:
        pytest.skip(f"states after layers {a} and {b} not both saved")
    ck = _checkpoint_or_skip([b], extra=("model.embed_tokens.weight",) if a < 0 else ())
    if a < 0:  # the stored embedding is the checkpoint's embedding rows of the ids
        W = ck.get("model.embed_tokens.weight")
        for p in golden.prompts:
            assert torch.equal(x_in[p.name][:, :, 0], W[torch.tensor([p.ids])]), p.name
    layer = _ref_layer(ck.args(), b)
    try:
        with torch.no_grad():
            for p in golden.prompts:
                x = x_in[p.name]
                y = layer(x, _positions(x), None, "expanded")
                assert torch.equal(y, x_out[p.name]), (p.name, b, _max_abs(y, x_out[p.name]), pcc(y, x_out[p.name]))
    finally:
        del layer
        gc.collect()


# =================================================================================================
# 7. stored states vs the official HF decoder layers
# =================================================================================================
HF_CHAINS = [(-1, 0), (4, 7), (7, 8), (32, 35), (47, 51), (51, 52)]


@pytest.fixture(scope="module")
def hf():
    if not (DEFAULT_WEIGHTS_DIR / "config.json").exists() or not (HF_META_DIR / "modeling_motif.py").exists():
        pytest.skip("HF sources / config not available")
    cfg_mod, mm = load_hf_modules()
    cfg = hf_config_from_json(DEFAULT_WEIGHTS_DIR, cfg_mod)
    with hf_cpu_flash_attention():
        rotary = mm.MotifRotaryEmbedding(cfg, rope_head_dim=cfg.qk_rope_head_dim)
    return SimpleNamespace(mm=mm, cfg=cfg, rotary=rotary)


def _hf_run_layers(hf, layers, xs: dict) -> dict:
    with hf_cpu_flash_attention(), torch.no_grad():
        for L in layers:
            layer = _hf_layer(hf.mm, hf.cfg, L)
            xs = {n: _hf_apply(layer, hf.rotary, x) for n, x in xs.items()}
            del layer
            gc.collect()
    return xs


@pytest.mark.parametrize("chain", HF_CHAINS, ids=[f"{a}to{b}" for a, b in HF_CHAINS])
def test_production_states_match_hf_decoder_layers(golden, hf, chain):
    a, b = chain
    x_out = _stored_state(golden, b)
    if x_out is None or (a >= 0 and not state_path(golden.dir, a).exists()):
        pytest.skip(f"states after layers {a} and {b} not both saved")
    layers = list(range(a + 1, b + 1))
    ck = _checkpoint_or_skip(layers, extra=("model.embed_tokens.weight",) if a < 0 else ())
    if a < 0:  # the layer-0 input from the ids, as HF MotifModel.forward builds it
        W = ck.get("model.embed_tokens.weight")
        stored_emb = load_tensors(golden.dir / "states" / "embed.safetensors")[0]
        xs = {}
        for p in golden.prompts:
            h = F.embedding(torch.tensor([p.ids]), W)
            assert torch.equal(h, stored_emb[p.name]), p.name
            xs[p.name] = h.unsqueeze(2).expand(-1, -1, E, -1).contiguous()
    else:
        xs = load_tensors(state_path(golden.dir, a))[0]
    ys = _hf_run_layers(hf, layers, xs)
    for n, y in ys.items():
        assert torch.equal(y, x_out[n]), (n, layers, _max_abs(y, x_out[n]), pcc(y, x_out[n]))
    print(
        f"\nHF layers {layers[0]}..{layers[-1]} on the stored state after {a}: bit-identical to the stored state after {b}"
    )


@pytest.mark.skipif(
    os.environ.get("MOTIF3_VERIFY_FULL_CHAIN") != "1", reason="set MOTIF3_VERIFY_FULL_CHAIN=1 (~2.5 min)"
)
def test_production_full_chain_hf(golden, hf):
    """ids -> HF embedding -> HF layers 0..last -> HF head; every saved state, the resume state and each head."""
    last = golden.manifest["last_layer"]
    ck = _checkpoint_or_skip(range(last + 1), extra=("model.embed_tokens.weight",))
    W = ck.get("model.embed_tokens.weight")
    xs = {
        p.name: F.embedding(torch.tensor([p.ids]), W).unsqueeze(2).expand(-1, -1, E, -1).contiguous()
        for p in golden.prompts
    }
    del W
    head_local = all(ck.is_local(n) for n in ("model.norm.weight", "lm_head.weight"))
    bad, n_saved, n_heads = [], 0, 0
    for L in range(last + 1):
        xs = _hf_run_layers(hf, [L], xs)
        stored = _stored_state(golden, L)
        if stored is not None:
            n_saved += 1
            bad += [(L, n, _max_abs(xs[n], stored[n])) for n in xs if not torch.equal(xs[n], stored[n])]
        if head_local and str(L) in golden.manifest["heads"]:
            _check_head(golden, hf, L, xs)
            n_heads += 1
    assert not bad, bad
    print(f"\nHF full chain 0..{last}: {n_saved} stored states and {n_heads} head(s) bit-identical")


# =================================================================================================
# 8. heads
# =================================================================================================
def _check_head(golden, hf, L: int, xs: dict) -> None:
    ck = MotifCheckpoint(DEFAULT_WEIGHTS_DIR)
    norm = hf.mm.MotifRMSNorm(D, eps=hf.cfg.rms_norm_eps)
    norm.weight = torch.nn.Parameter(ck.get("model.norm.weight"), requires_grad=False)
    W = ck.get("lm_head.weight")
    lt = load_tensors(head_paths(golden.dir, L)[0])[0]
    ht = load_tensors(head_paths(golden.dir, L)[1])[0]
    summary = golden.manifest["heads"][str(L)]["summary"]
    for p in golden.prompts:
        n, S = p.name, len(p.ids)
        with torch.no_grad():
            sm = xs[n].mean(dim=2)
            h = norm(sm)
            logits = F.linear(h, W).float()[0]
        assert torch.equal(ht[f"{n}.stream_mean"], sm.float()) and torch.equal(ht[f"{n}.final_hidden"], h.float()), n
        assert torch.isfinite(logits).all(), n
        top_v, top_i = torch.topk(logits, TOPK, dim=-1)
        assert torch.equal(lt[f"{n}.topk_ids"], top_i) and torch.equal(lt[f"{n}.topk_logits"], top_v), n
        assert torch.equal(lt[f"{n}.argmax"], logits.argmax(-1))
        assert torch.equal(lt[f"{n}.argmax_ties"], (logits == logits.max(-1, keepdim=True).values).sum(-1))
        lse = torch.logsumexp(logits, -1)
        assert torch.equal(lt[f"{n}.logsumexp"], lse)
        tgt = torch.tensor(p.ids[1:])
        assert torch.equal(lt[f"{n}.target_ids"], torch.tensor(p.ids[1:] + [-1]))
        t_logit = logits[torch.arange(S - 1), tgt]
        assert torch.equal(lt[f"{n}.target_logit"][:-1], t_logit) and torch.isnan(lt[f"{n}.target_logit"][-1])
        logp = torch.log_softmax(logits, -1)[torch.arange(S - 1), tgt]
        torch.testing.assert_close(lt[f"{n}.target_logprob"][:-1], logp, atol=2e-4, rtol=0)
        rank = (logits[:-1] > t_logit[:, None]).sum(-1)
        assert torch.equal(lt[f"{n}.target_rank"][:-1], rank) and lt[f"{n}.target_rank"][-1] == -1
        in_asst = torch.tensor([r in (3, 4) for r in p.roles[1:]] + [False])
        assert torch.equal(lt[f"{n}.target_in_assistant"], in_asst)
        # the manifest's summary metrics
        lp = lt[f"{n}.target_logprob"][:-1]
        for region, mask in (("all", torch.ones(S - 1, dtype=torch.bool)), ("assistant", in_asst[:-1])):
            got = summary[n][region]
            assert got["n"] == int(mask.sum())
            if got["n"]:
                r = rank[mask]
                assert got["top1"] == round(float((r == 0).float().mean()), 4)
                assert got["top5"] == round(float((r < 5).float().mean()), 4)
                assert abs(got["nll"] - float(-lp[mask].mean())) <= 1e-4
        assert summary[n]["last_position_top5_ids"] == top_i[-1, :5].tolist() and summary[n]["finite"]


def test_production_heads_recomputed_from_stored_states(golden, hf):
    if not golden.manifest["heads"]:
        pytest.skip("no head in the production golden yet")
    _checkpoint_or_skip(extra=("model.norm.weight", "lm_head.weight"))
    for k in golden.manifest["heads"]:
        xs = _stored_state(golden, int(k))
        if xs is None:
            pytest.skip(f"state after layer {k} not saved")
        _check_head(golden, hf, int(k), xs)


# =================================================================================================
# 9. sanity of the stored states
# =================================================================================================
def _token_rms(x: torch.Tensor) -> torch.Tensor:
    """Per-token rms over the 4 streams ``[S]`` (fp32)."""
    return x[0].float().pow(2).mean(dim=tuple(range(1, x.dim() - 1))).sqrt()


def test_production_states_sanity(golden):
    saved = golden.manifest["saved_layers"]
    if not saved:
        pytest.skip("no saved states")
    names = [p.name for p in golden.prompts]
    med = {n: [] for n in names}
    last_states = None
    for L in saved:
        st = load_tensors(state_path(golden.dir, L))[0]
        for n in names:
            assert torch.isfinite(st[n].float()).all(), (n, L)
            med[n].append(float(_token_rms(st[n]).median()))
        if L == saved[-1]:
            last_states = st
    print("\nmedian token rms per saved layer " + str(saved))
    for n in names:
        s = med[n]
        print(f"  {n:18s} " + " ".join(f"{v:.3f}" for v in s))
        # layers 0-35: 0.04 -> 0.26..0.38, at most x1.8 per saved step (loose bounds leave room for layers 36-52)
        assert all(0.005 < v < 10.0 for v in s), (n, s)
        assert all(b >= 0.85 * a for a, b in zip(s, s[1:])), (n, s)  # no collapse
        assert all(b <= 4.0 * a for a, b in zip(s, s[1:])), (n, s)  # no explosion of typical tokens
        if saved[0] == 0 and saved[-1] >= 32:
            assert 2.0 < s[-1] / s[0] < 100.0, (n, s)

    # tokens far above the typical scale at the last saved layer (attention sinks, massive activations)
    report = []
    for p in golden.prompts:
        r = _token_rms(last_states[p.name])
        for i in torch.nonzero(r > 50 * r.median()).flatten().tolist():
            report.append(f"{p.name}[{i}] id {p.ids[i]} rms {float(r[i]):.1f} (median {float(r.median()):.3f})")
    print(f"tokens with rms > 50x median after layer {saved[-1]}: {report or 'none'}")

    # causality: a shared prompt prefix gives the same states (up to bf16 GEMM-shape rounding)
    for L in (saved[0], saved[-1]):
        st = load_tensors(state_path(golden.dir, L))[0]
        for pa, pb in itertools.combinations(golden.prompts, 2):
            k = next((i for i, (u, v) in enumerate(zip(pa.ids, pb.ids)) if u != v), min(len(pa.ids), len(pb.ids)))
            if k == 0:
                continue
            xa, xb = st[pa.name][0, :k].float(), st[pb.name][0, :k].float()
            rel = float((xa - xb).norm() / xa.norm())
            assert rel < 3e-2, (L, pa.name, pb.name, k, rel)

    # bf16 tracks the fp32 sanity run (same prompts, layers 0..7), outliers included
    m32 = FP32_SANITY_DIR / "manifest.json"
    if m32.exists():
        man32 = json.loads(m32.read_text(encoding="utf-8"))
        assert (
            man32["mode"]["dtype"] == "fp32"
            and man32["prompt_set"]["sha256"] == golden.manifest["prompt_set"]["sha256"]
        )
        for L in man32["saved_layers"]:
            if not state_path(golden.dir, L).exists():
                continue
            b16 = load_tensors(state_path(golden.dir, L))[0]
            f32 = load_tensors(state_path(FP32_SANITY_DIR, L))[0]
            for n in names:
                xb, xf = b16[n].float(), f32[n]
                rel = float((xb - xf).norm() / xf.norm())
                assert pcc(xb, xf) > 0.9995 and rel < 3e-2, (L, n, pcc(xb, xf), rel)
                rb, rf = _token_rms(b16[n]), _token_rms(f32[n])
                big = rf > 50 * rf.median()
                if big.any():
                    assert torch.allclose(rb[big], rf[big], rtol=2e-2), (L, n, rb[big], rf[big])
            print(
                f"bf16 vs fp32 sanity run after layer {L}: min pcc {min(pcc(b16[n].float(), f32[n]) for n in names):.6f}"
            )
