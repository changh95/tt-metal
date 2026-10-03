# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Work package 3 of the features design (``docs/features/FEATURES_DESIGN.md`` §3.6, §4 gate G14, §5.2; README §17):
the MTP layer ``tt/mtp.py`` (``model.mtp_layers.0``), its embedding and LM-head hooks
(``MotifEmbedding.embed_rows`` / ``embed_rows_from_device``, ``MotifLMHead.decode_logits``), the MTP weight-name
helpers of ``tt/weights.py`` and the TT-cache part ``L53`` (``scripts/convert_weights.py --mtp``).

Host only (root conftest active, devices hidden; the ``-k cpu`` tests; the first run builds the reference goldens under
``tt_cache/test/mtp`` in ~2 min)::

    S=/home/ttuser/hchang/experiments/motif-3/scripts
    $S/hostrun.sh -n mtp_host -- python -m pytest -p no:cacheprovider -q models/demos/motif3/tests/unit/test_mtp.py \
        -k cpu

Device (gate G14 and the acceptance estimate; real weights from the TT cache: the globals and part ``L53``, loaded
with a weight source that raises on any read)::

    $S/devrun.sh -t 2400 -n mtp -- python -m pytest models/demos/motif3/tests/unit/test_mtp.py -k "not cpu" -s \
        -p no:cacheprovider

Goldens (:func:`make_mtp_goldens`): the reference ``MotifMTP`` (``reference/modules.py``) with the checkpoint's MTP
tensors (shard 104), run teacher-forced over the 6 C2 prompts as the reference defines it: row ``p`` = ``(hn_p,
embed(t_{p+1}))`` at position ``p``, ``hn`` = the C2 golden's post-final-norm hidden states, 2965 rows. Two copies: fp32
(bf16-valued weights and inputs; the "ideal" golden every TT comparison uses) and bf16 (the numerics of the CPU
acceptance estimate of ``docs/features/spec_mtp.md`` §1.2: 0.760 teacher-forced, 0.835 on-policy). The acceptance of
row ``p`` is ``m_p == argmax_main[p + 1]`` (the main model's choice for ``t_{p+2}``, i.e. the plugin's greedy walk);
"on-policy" rows have ``t_{p+1} == argmax_main[p]``.

Every device check prints an ``[mtp]`` line. Thresholds (G14; README §12): GDLA >= 0.999 per lane, MLP >= 0.9995 (on the
TT's own MLP input), MTP output >= 0.995, MTP cache vs the reference ``c_kv / gamma`` (and ``k_pe``) >= 0.9999 (every
single decode-written row >= :data:`LAT_ROW_MIN`), MTP argmax == the fp32 reference on >= 99 % of the rows without a
bf16 tie; every comparison also needs a relative error below :data:`REL_MAX` (PCC is 1.0 when one side is constant).

Decode KV writes are checked against poisoned slots: the KV-only history fill writes every decode slot ``p`` (and
``p + 1``) from the decode's own inputs, so the tests overwrite those slots with stale rows before a decode step and
then compare all 32 chips' cache copies with the ``tt/kv_write.py`` host model of the step (``apply_kv_writes_host``:
the slots the mode writes, on the chips it writes, nothing else). :func:`test_mtp_decode_kv_writes` covers inactive
lanes, DP-row relocation, ``kv_write`` modes ``all`` / ``all_split`` (packed verify with partners on other DP rows) and
a partner row vs the same row computed by an ordinary step.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.test_attention import (
    HF_META,
    MESH,
    _Capture,
    _decode_step_inputs,
    _dev,
    _free,
    _free_step,
    _gather_cache,
    _host_quant,
    _lane_x,
    _per_lane_out,
    _replicated,
    _rows,
    _setup,
    _traced_us,
    _upload_cache,
    fmt,
    pcc,
    random_attn_tensors,
    ref_args,
    ref_latents,
    stats,
)
from models.demos.motif3.tests.unit.test_attention_resumed import _rmsnorm, emulate_attention
from models.demos.motif3.tt import kv_write as KW
from models.demos.motif3.tt import mtp as MTPM
from models.demos.motif3.tt import weights as W
from models.demos.motif3.tt.attention import MTP_ATTN_PREFIX, MotifAttention, _AttnSource
from models.demos.motif3.tt.model_config import PROJECT_ROOT, MotifTTConfig

TEST_DIR = PROJECT_ROOT / "tt_cache" / "test" / "mtp"
C2_DIR = PROJECT_ROOT / "goldens" / "c2"
GOLDENS = TEST_DIR / "mtp_goldens_c2.pt"
GOLDENS_FORMAT = 1  # bump when make_mtp_goldens changes meaning
C2_PROMPTS = ("chat_default", "en_technical", "ko_passage", "math_word_problem", "python_code", "multi_turn_chat")
DECODE_PROMPTS = ("en_technical", "ko_passage", "python_code", "math_word_problem")  # one per DP row (>= 527 rows)
# decode positions per DP row: past the 129-key window, block edges (63/64 of a block), distinct within a row, < 512
DECODE_POS = [
    [129, 191, 192, 255, 256, 300, 383, 448],
    [136, 160, 200, 263, 320, 384, 447, 500],
    [143, 175, 224, 287, 319, 352, 415, 510],
    [130, 150, 230, 290, 330, 400, 460, 480],
]  # fmt: skip
HIST = 512  # history rows filled per decode lane (one 512 bucket; > every decode position + 1, the draft slot)
REPO_ROOT = Path(__file__).resolve().parents[5]  # tt-metal
HEAD_COMMIT = "HEAD"  # the committed lm_head.py / embedding.py the op-sequence tests compare against
OUT_PCC_MIN = 0.995  # G14: MTP output (final_layernorm), aggregate and per decode lane
ATTN_PCC_MIN = 0.999  # G14: GDLA output, aggregate and per lane
MLP_PCC_MIN = 0.9995  # G14: dense PolyNorm MLP on its own input
LAT_PCC_MIN = 0.9999  # G14: MTP cache rows vs the reference c_kv / gamma and k_pe
LAT_ROW_MIN = 0.999  # every single decode-written row (a row written from another lane's input has PCC ~0)
ARGMAX_AGREE_MIN = 0.99  # G14: MTP argmax == the fp32 reference's, rows without a bf16 tie
ALPHA_CPU = {"all": 0.760, "on_policy": 0.835}  # spec_mtp.md §1.2 (bf16 reference, 2965 / 1970 rows)
ALPHA_TOL = 0.02  # G-S4's bar ("within 2 points"), applied to the TT MTP on the golden hidden states
REL_MAX = 0.1


def log(msg: str) -> None:
    print(f"[mtp] {msg}", flush=True)


def host_cfg(**kw) -> MotifTTConfig:
    return MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8), **kw)


def _good(s: Dict[str, float], pcc_min: float) -> bool:
    return s["pcc"] >= pcc_min and s["rel_fro"] < REL_MAX and s["nonfinite"] == 0


# ======================================================================================================================
# weights and the reference
# ======================================================================================================================
def real_mtp_tensors(dtype=torch.float32) -> Dict[str, torch.Tensor]:
    """The 19 checkpoint tensors ``model.mtp_layers.0.*`` (shard 104) in ``dtype`` (HF names)."""
    try:
        loader = W.HFWeightLoader()
        if not W.mtp_layer_available(loader):
            pytest.skip("the MTP layer's tensors are not on disk (checkpoint shard 104)")
        return {n: loader.get(n).to(dtype) for n in W.mtp_tensor_names()}
    except (W.MissingWeightError, FileNotFoundError) as e:
        pytest.skip(f"MTP weights not on disk: {e}")


def random_mtp_tensors(args, seed: int) -> Dict[str, torch.Tensor]:
    """Real-dim random MTP weights (HF names; ``reference.weights.random_state_dict`` statistics), bf16-valued fp32."""
    g = torch.Generator().manual_seed(seed)
    D, I = args.hidden_size, args.intermediate_size

    def linear(o, i):
        return torch.randn(o, i, generator=g) * i**-0.5

    t = {W.mtp_name(f"self_attn.{k}.weight"): v for k, v in random_attn_tensors(args, seed).items()}
    t[W.mtp_name("mlp.gate_proj.weight")] = linear(I, D)
    t[W.mtp_name("mlp.up_proj.weight")] = linear(I, D)
    t[W.mtp_name("mlp.down_proj.weight")] = linear(D, I)
    t[W.mtp_name("mlp.act_fn.weight")] = torch.randn(3, generator=g)
    t[W.mtp_name("mlp.act_fn.bias")] = torch.rand(1, generator=g) * 2 - 1
    t[W.mtp_name("input_proj.weight")] = linear(D, 2 * D)
    for n in W.MTP_NORMS:
        t[W.mtp_name(f"{n}.weight")] = torch.rand(D, generator=g) + 0.5
    return {k: v.to(torch.bfloat16).float() for k, v in t.items()}


def ref_mtp(args, tensors: Dict[str, torch.Tensor], dtype):
    """Reference ``MotifMTP`` with the HF-named ``tensors`` in ``dtype``."""
    from models.demos.motif3.reference.modules import MotifMTP

    p = W.mtp_name() + "."
    with torch.device("meta"):
        m = MotifMTP(args)
    m.load_state_dict({k[len(p) :]: v.to(dtype) for k, v in tensors.items()}, strict=True, assign=True)
    return m.eval().requires_grad_(False)


def ref_steps(mtp, h_main: torch.Tensor, e_next: torch.Tensor, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """``MotifMTP.forward`` op for op (``[B, S, D]`` inputs), keeping every intermediate."""
    e = mtp.embed_norm(e_next)
    h = mtp.input_proj(torch.cat([h_main, e], dim=-1))
    a = mtp.input_layernorm(h)
    o = mtp.self_attn(a, positions)
    h1 = h + o
    f = mtp.post_attention_layernorm(h1)
    u = mtp.mlp(f)
    h2 = h1 + u
    return dict(e_norm=e, h=h, a=a, o=o, h_mid=h1, f=f, u=u, h_out=h2, out=mtp.final_layernorm(h2))


# ======================================================================================================================
# host emulation of the TT dataflow (exact algebra, fp64)
# ======================================================================================================================
def emulate_dense_mlp(cfg: MotifTTConfig, src, prefix: str, x: torch.Tensor, layer: int) -> torch.Tensor:
    """``PolyNormMLP(kind="dense")``'s per-chip TP math with the module's transforms (``tests/unit/test_mlp.py``
    ``test_tp_shards_reproduce_reference_mlp``): column-parallel gate | up, moments summed over the 8 chips, Horner,
    row-parallel down with the output scale folded, partials summed."""
    from models.demos.motif3.tt.polynorm import PolyNormCoefficients, horner_scale_constants, polynorm_output_scale

    gw, uw, dw = (src.get(f"{prefix}.{k}_proj.weight").to(x.dtype) for k in ("gate", "up", "down"))
    coeffs = PolyNormCoefficients.from_source(src, prefix)
    inter = gw.shape[0]
    D_, E_ = horner_scale_constants(torch.tensor(coeffs.by_power, dtype=torch.float64), inter, cfg.polynorm_eps,
                                    dtype=x.dtype)  # fmt: skip
    n = inter // cfg.tp
    scale = polynorm_output_scale(cfg, layer)
    gus = [W.mlp_gate_up_for_chip(gw, uw, cfg, t) for t in range(cfg.tp)]
    gs = [x @ gu[:, :n] for gu in gus]
    us = [x @ gu[:, n:] for gu in gus]
    s_ = torch.stack([sum(g.pow(2 * k).sum(-1) for g in gs) for k in (1, 2, 3)], dim=-1)  # the TP all-reduce
    a_ = torch.rsqrt(s_ * D_ + E_)
    y = 0
    for t in range(cfg.tp):
        g = gs[t]
        h = (((a_[:, 2:3] * g + a_[:, 1:2]) * g + a_[:, 0:1]) * g + coeffs.b) * us[t]
        y = y + h @ W.mlp_down_for_chip(dw, cfg, t, output_scale=scale)
    return y


def emulate_mtp(cfg: MotifTTConfig, src, hn: torch.Tensor, e: torch.Tensor, mode: str) -> Dict[str, torch.Tensor]:
    """``MotifMTP``'s per-chip dataflow at positions ``0 .. T-1`` (math in ``hn``'s dtype): ``hn [T, D]`` (main
    post-norm hidden), ``e [T, D]`` (embedding rows of ``t_{p+1}``). The interleaved K split of ``input_proj``
    (``weights.mtp_input_proj_rows``: the chip's slice is ``cat[partition(hn), partition(e_norm)]``), the norms with
    their gammas, the attention (``test_attention_resumed.emulate_attention``, ``mode`` "decode" = absorbed MQA over the
    latent, "prefill" = expanded GQA) and the dense MLP (:func:`emulate_dense_mlp`)."""
    eps, L = cfg.rms_norm_eps, cfg.mtp_layer_idx
    g = {n: src.get(W.mtp_name(f"{n}.weight")).to(hn.dtype) for n in W.MTP_NORMS}

    def norm(x, name):
        return _rmsnorm(x, eps) * g[name]

    en = norm(e, "embed_norm")
    w_in = src.get(W.mtp_name("input_proj.weight")).to(hn.dtype)
    k = cfg.hidden_size // cfg.tp
    h = 0
    for tp in range(cfg.tp):
        x_in = torch.cat([hn[:, k * tp : k * (tp + 1)], en[:, k * tp : k * (tp + 1)]], -1)  # partition + concat
        h = h + x_in @ W.mtp_input_proj_for_chip(w_in, cfg, tp)
    a = norm(h, "input_layernorm")
    o = emulate_attention(cfg, cfg.mtp_layer_spec(), _AttnSource(src, L, W.mtp_name("self_attn")), a, mode)
    h1 = h + o
    f = norm(h1, "post_attention_layernorm")
    u = emulate_dense_mlp(cfg, src, W.mtp_name("mlp"), f, L)
    h2 = h1 + u
    return dict(e_norm=en, h=h, a=a, o=o, h_mid=h1, f=f, u=u, h_out=h2, out=norm(h2, "final_layernorm"))


# ======================================================================================================================
# reference goldens (host; cached)
# ======================================================================================================================
def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of bf16 numbers at ``|x|`` (8 significant bits)."""
    e = torch.floor(torch.log2(x.abs().clamp_min(1e-30)))
    return torch.pow(2.0, e - 7)


@torch.no_grad()
def make_mtp_goldens(path: Path = GOLDENS) -> Path:
    """Host only. The reference ``MotifMTP`` over the 6 C2 prompts (module docstring), concatenated row by row
    (``rows[name] = (offset, n)``, ``n = S - 1``). Saves per row: the inputs ``hn`` / ``e_next`` (bf16 values) and
    ``next_ids`` (``t_{p+1}``), the fp32 reference's ``a`` (attention input), ``o`` (attention output), ``u`` (MLP
    output), ``out`` (final norm) and its LM-head argmax ``m32`` / top-2 margin, the bf16 reference's argmax ``m16``,
    the acceptance target ``argmax_main[p + 1]``, ``on_policy`` and ``tie`` (a bf16 tie of the main logits at ``p`` or
    ``p + 1``). ~0.25 GB."""
    from safetensors import safe_open

    args = ref_args()
    t32 = real_mtp_tensors(torch.float32)
    m32 = ref_mtp(args, t32, torch.float32)
    m16 = ref_mtp(args, t32, torch.bfloat16)
    lm16 = W.HFWeightLoader().get("lm_head.weight")  # [V, D] bf16
    lm32 = lm16.float()
    prompts = {p["name"]: p for p in json.loads((C2_DIR / "prompts.json").read_text())["prompts"]}
    keys = ("hn", "e_next", "next_ids", "a", "o", "u", "out", "m32", "top1_32", "margin32", "m16", "target")
    keys += ("on_policy", "tie")
    cols: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}
    rows, ids_all, off = {}, {}, 0
    t0 = time.time()
    with (
        safe_open(str(C2_DIR / "final" / "hidden_after_layer_52.safetensors"), "pt") as fh,
        safe_open(str(C2_DIR / "states" / "embed.safetensors"), "pt") as fe,
        safe_open(str(C2_DIR / "final" / "logits_after_layer_52.safetensors"), "pt") as fl,
    ):
        for name in C2_PROMPTS:
            ids = torch.tensor(prompts[name]["ids"], dtype=torch.long)
            S = int(ids.numel())
            hn = fh.get_tensor(f"{name}.final_hidden")[0].to(torch.bfloat16)  # [S, D] bf16 values (post final norm)
            emb = fe.get_tensor(name)[0]  # [S, D] bf16 = embed(ids)
            am = fl.get_tensor(f"{name}.argmax")
            ties = fl.get_tensor(f"{name}.argmax_ties")
            n = S - 1
            pos = torch.arange(n)[None]
            st = {k: v[0] for k, v in ref_steps(m32, hn[None, :n].float(), emb[None, 1:].float(), pos).items()}
            full = m32(hn[None, :n].float(), emb[None, 1:].float(), pos)[0]
            assert torch.equal(full, st["out"]), "ref_steps must be MotifMTP.forward"
            tops, args32 = [], []
            for i in range(0, n, 256):
                lg = st["out"][i : i + 256] @ lm32.t()
                tops.append(lg.topk(2, dim=-1).values)
                args32.append(lg.argmax(-1))  # torch.argmax: the lowest index of the maxima (the device tie rule)
            top, m32_ids = torch.cat(tops), torch.cat(args32)
            out16 = m16(hn[None, :n], emb[None, 1:], pos)[0]
            m16_ids = torch.cat([torch.nn.functional.linear(out16[i : i + 256], lm16).float().argmax(-1)
                                 for i in range(0, n, 256)])  # fmt: skip
            cols["hn"].append(hn[:n])
            cols["e_next"].append(emb[1:])
            cols["next_ids"].append(ids[1:])
            for k in ("a", "o", "u", "out"):
                cols[k].append(st[k].contiguous())
            cols["m32"].append(m32_ids)
            cols["top1_32"].append(top[:, 0])
            cols["margin32"].append(top[:, 0] - top[:, 1])
            cols["m16"].append(m16_ids)
            cols["target"].append(am[1:])
            cols["on_policy"].append(ids[1:] == am[:-1])
            cols["tie"].append((ties[1:] > 1) | (ties[:-1] > 1))
            rows[name] = (off, n)
            ids_all[name] = ids
            off += n
            log(f"goldens: {name} {n} rows ({time.time() - t0:.1f} s)")
    d = {k: torch.cat(v) for k, v in cols.items()}
    d.update(format=GOLDENS_FORMAT, rows=rows, ids=ids_all, prompts=list(C2_PROMPTS))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, path)
    return path


def load_goldens() -> Dict[str, Any]:
    """:func:`make_mtp_goldens` output (built on first use; skips when the C2 golden or shard 104 is missing)."""
    if GOLDENS.is_file():
        d = torch.load(GOLDENS, weights_only=False)
        if d.get("format") == GOLDENS_FORMAT:
            return d
    try:
        make_mtp_goldens()
    except (FileNotFoundError, KeyError, W.MissingWeightError) as e:
        pytest.skip(f"cannot build the MTP goldens: {e}")
    return torch.load(GOLDENS, weights_only=False)


def acceptance(m: torch.Tensor, gold: Dict[str, Any], rows: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Acceptance rates of drafts ``m`` (per golden row): all rows, on-policy rows, on-policy rows without a tie."""
    sel = torch.ones(m.numel(), dtype=torch.bool) if rows is None else rows
    hit = (m.long() == gold["target"]) & sel
    onp = gold["on_policy"] & sel
    clean = onp & ~gold["tie"]

    def rate(num, den):
        return float(num.sum()) / max(int(den.sum()), 1)

    return {
        "all": rate(hit, sel),
        "on_policy": rate(hit & onp, onp),
        "on_policy_no_tie": rate(hit & clean, clean),
        "n": int(sel.sum()),
        "n_on_policy": int(onp.sum()),
    }


def non_tie_rows(gold: Dict[str, Any]) -> torch.Tensor:
    """Rows whose fp32 reference MTP logits have a top-2 margin of at least two bf16 spacings (no bf16 tie)."""
    return gold["margin32"] >= 2 * _bf16_ulp(gold["top1_32"])


# ======================================================================================================================
# a recording fake ttnn (op-sequence tests)
# ======================================================================================================================
class _FT:
    """Fake tensor: shape + a serial id (lineage in the recorded calls)."""

    _n = 0

    def __init__(self, shape: Sequence[int], layout: str = "TILE"):
        _FT._n += 1
        self.id, self.shape, self.layout = _FT._n, tuple(int(s) for s in shape), layout


class _FakeTTNN(ModuleType):
    """Records every call as ``(op, arg descriptors, sorted kwarg names)``; returns fresh fake tensors with the
    shapes of the real ops (only the ops the head / embedding decode paths use)."""

    TILE_LAYOUT, ROW_MAJOR_LAYOUT = "TILE", "ROW_MAJOR"
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = "DRAM", "L1"
    bfloat16, float32, int32, uint32 = "bf16", "fp32", "i32", "u32"
    Tensor = _FT

    def __init__(self):
        super().__init__("ttnn")
        self.calls: List[tuple] = []

    @staticmethod
    def _d(v):
        if isinstance(v, _FT):
            return ("T", v.id, v.shape)
        if isinstance(v, (list, tuple)):
            return tuple(_FakeTTNN._d(x) for x in v)
        return repr(v)

    def _rec(self, op, args, kw, shape=None, layout="TILE"):
        self.calls.append((op, tuple(self._d(a) for a in args), tuple(sorted(kw))))
        return None if shape is None else _FT(shape, layout)

    def deallocate(self, t, *a):
        self._rec("deallocate", (t,), {})

    def sum(self, x, dim, keepdim=True, **kw):
        s = list(x.shape)
        s[dim] = 1
        return self._rec("sum", (x, dim), kw, s)

    def rms_norm(self, x, **kw):
        return self._rec("rms_norm", (x,), kw, x.shape)

    def to_memory_config(self, x, mc):
        return self._rec("to_memory_config", (x, mc), {}, x.shape, x.layout)

    def linear(self, x, w, **kw):
        return self._rec("linear", (x, w), kw, x.shape[:-1] + (w.shape[-1],))

    def untilize(self, x, **kw):
        return self._rec("untilize", (x,), kw, x.shape, "ROW_MAJOR")

    def reshape(self, x, shape):
        return self._rec("reshape", (x, tuple(shape)), {}, tuple(shape), x.layout)

    def embedding(self, tok, w, **kw):
        return self._rec("embedding", (tok, w), kw, tok.shape + (w.shape[-1],))

    def repeat(self, x, reps, **kw):
        return self._rec("repeat", (x, tuple(reps)), kw, tuple(a * b for a, b in zip(x.shape, reps)), x.layout)


class _FakeCCL:
    def __init__(self, fake: _FakeTTNN, dp: int = 4):
        self.fake, self.dp = fake, dp

    def ag_dp_rows(self, x, **kw):
        return self.fake._rec("ccl.ag_dp_rows", (x,), kw, x.shape[:-2] + (x.shape[-2] * self.dp, x.shape[-1]))

    def partition(self, x, dim, axis, **kw):
        s = list(x.shape)
        s[dim] //= self.dp if axis == "dp" else 8
        return self.fake._rec(f"ccl.partition.{axis}", (x, dim), kw, s, x.layout)

    def all_gather(self, x, dim, axis, **kw):
        s = list(x.shape)
        s[dim] *= 8
        return self.fake._rec(f"ccl.all_gather.{axis}", (x, dim), kw, s, x.layout)


def _git_module(rel: str, name: str):
    """``rel`` (a motif3 file) as committed at :data:`HEAD_COMMIT`, imported as ``name`` with the package context of
    the current one (its relative imports resolve to the current shared infra). Skips without git."""
    try:
        res = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{HEAD_COMMIT}:{rel}"], capture_output=True,
                             text=True, check=True)  # fmt: skip
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"git show {HEAD_COMMIT}:{rel} failed: {e}")
    pkg = rel.rsplit("/", 1)[0].replace("/", ".")
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    mod.__file__ = str(REPO_ROOT / rel)
    exec(compile(res.stdout, mod.__file__, "exec"), mod.__dict__)
    return mod


def _fake_head(mod, fake: _FakeTTNN, vocab_split: str, stream_reduce: str = "sum"):
    """A ``MotifLMHead`` of module ``mod`` without ``__init__`` (no device), wired to the fake ttnn."""
    cfg = host_cfg()
    h = object.__new__(mod.MotifLMHead)
    h.cfg, h.ccl, h.vocab_split, h.stream_reduce = cfg, _FakeCCL(fake), vocab_split, stream_reduce
    h.n_streams, h.hidden, h.lanes = 4, cfg.hidden_size, cfg.lanes_per_row
    h.vc = mod.vocab_per_shard(cfg, vocab_split)
    h.logits_dtype, h.memory_config, h.eps, h.norm_eps = "bf16", "DRAM", 1e-5, 1.6e-4
    h.ckc_norm, h.ckc_lm, h.pc = "ckc_norm", "ckc_lm", "pc_lm"
    h.norm_grid, h._norm_sharded, h._w_mean = (8, 4), ("norm_mc", "norm_pc"), {}
    h.gamma, h.weight = _FT((1, 1, 128, 32), "ROW_MAJOR"), _FT((1, 1, 4096, h.vc))
    return h


# ======================================================================================================================
# host tests
# ======================================================================================================================
def test_cpu_mtp_names_and_index(tmp_path):
    """The weight-name helpers list exactly the checkpoint's 19 MTP tensors (all in shard 104, on disk), the attention
    prefix is the one ``tt/attention.py`` uses, and the layer identity is the MTP layer's (index 53, part L53, SWA 129,
    plain RoPE, scale 192^-0.5, dense MLP with output scale 0.5). Availability needs all 19 tensors: a source that
    lists only some of them is refused by ``weights.mtp_layer_available`` and ``mtp.mtp_weights_available`` (it would
    fail at load time)."""
    from models.demos.motif3.tt.polynorm import polynorm_output_scale

    cfg = host_cfg()
    names = W.mtp_tensor_names()
    assert len(names) == 19 and len(set(names)) == 19
    assert W.mtp_name() == "model.mtp_layers.0" and W.mtp_name("self_attn") == MTP_ATTN_PREFIX
    assert W.mtp_name("input_proj.weight", 1) == "model.mtp_layers.1.input_proj.weight"
    loader = W.HFWeightLoader()
    assert W.mtp_names_in(loader) == names, sorted(set(W.mtp_names_in(loader)) ^ set(names))
    assert {loader.weight_map[n] for n in names} == {"model-00104-of-00155.safetensors"}
    assert W.mtp_layer_available(loader) and MTPM.mtp_weights_available(cfg)
    assert loader.shape(W.mtp_name("input_proj.weight")) == (4096, 8192)
    assert not W.mtp_layer_available(W.DictWeightSource({})) and W.mtp_names_in(W.DictWeightSource({})) == []
    # a partial index: every proper subset of the 19 is refused (here: each single tensor missing), all 19 accepted
    full = {n: torch.zeros(1) for n in names}
    no_l53 = host_cfg(tt_cache_root=tmp_path)  # no TT-cache part L53 there: the source decides
    assert not MTPM.mtp_cache_complete(no_l53)
    assert W.mtp_layer_available(W.DictWeightSource(full))
    assert MTPM.mtp_weights_available(no_l53, W.DictWeightSource(full))
    for drop in names:
        partial = W.DictWeightSource({k: v for k, v in full.items() if k != drop})
        assert W.mtp_names_in(partial) == [n for n in names if n != drop]
        assert not W.mtp_layer_available(partial) and not MTPM.mtp_weights_available(no_l53, partial), drop
    other_idx = W.DictWeightSource({W.mtp_name(n.split(".", 3)[-1], 1): v for n, v in full.items()})
    assert W.mtp_layer_available(other_idx, 1) and not W.mtp_layer_available(other_idx)
    spec = cfg.mtp_layer_spec()
    assert (cfg.mtp_layer_idx, spec.idx, spec.window, spec.rope_kind, spec.is_moe) == (53, 53, 129, "plain", False)
    assert abs(spec.softmax_scale - 0.07216878) < 1e-8 and polynorm_output_scale(cfg, 53) == 0.5
    assert W.layer_cache_marker(cfg, 53).parent.name == "L53"
    assert MTPM.mtp_cache_complete(cfg) == W.layer_cache_marker(cfg, 53).is_file()
    truncated = host_cfg(num_layers=4)
    assert truncated.mtp_layer_idx == 53 and truncated.mtp_layer_spec() == spec
    log(f"MTP tensors: {len(names)} in shard 104; part L53 converted: {MTPM.mtp_cache_complete(cfg)}")


def test_cpu_input_proj_k_split():
    """The interleaved K split of ``input_proj``: the 8 chips' column sets partition the 8192 inputs exactly once, each
    chip's input equals ``cat[partition_tp(hn), partition_tp(e)]`` (the module's two slices + concat), and the 8 partial
    products sum to ``cat[hn, e] @ W^T`` (fp64, exact algebra; real and random weights). The blocks stack into the
    ``[8192, 4096]`` host tensor whose TP shard ``tp`` (``tp_dim=0``) is chip ``tp``'s block."""
    cfg = host_cfg()
    rows = [W.mtp_input_proj_rows(cfg, tp) for tp in range(cfg.tp)]
    assert sorted(sum(rows, [])) == list(range(8192)) and all(len(r) == 1024 for r in rows)
    g = torch.Generator().manual_seed(7)
    hn, e = (torch.randn(40, 4096, generator=g, dtype=torch.float64) for _ in range(2))
    x = torch.cat([hn, e], -1)
    for tp in range(cfg.tp):
        part = torch.cat([hn.chunk(cfg.tp, -1)[tp], e.chunk(cfg.tp, -1)[tp]], -1)
        assert torch.equal(part, x[:, rows[tp]])
    for kind, w in (("random", torch.randn(4096, 8192, generator=g, dtype=torch.float64) / 90.5),
                    ("real", real_mtp_tensors()[W.mtp_name("input_proj.weight")].double())):  # fmt: skip
        blocks = [W.mtp_input_proj_for_chip(w, cfg, tp) for tp in range(cfg.tp)]
        y = sum(x[:, rows[tp]] @ blocks[tp] for tp in range(cfg.tp))
        s = stats(x @ w.t(), y)
        log(f"input_proj K split ({kind}): {fmt(s)}")
        assert s["rel_fro"] < 1e-12
        stacked = W.stack_tp(lambda tp: W.mtp_input_proj_for_chip(w, cfg, tp), cfg, dim=0)
        assert stacked.shape == (8192, 4096) and all(
            torch.equal(c, b) for c, b in zip(stacked.chunk(cfg.tp, 0), blocks)
        )
        assert blocks[0].dtype == torch.float64  # the fp64 path stays fp64 (exactness tests); fp32 otherwise
    with pytest.raises(ValueError):
        W.mtp_input_proj_for_chip(torch.zeros(4096, 4096), cfg, 0)


@pytest.mark.parametrize("weights_kind", ["random", "real"])
def test_cpu_mtp_block_matches_reference_fp64(weights_kind):
    """The whole MTP block as the TT module computes it per chip (:func:`emulate_mtp`: K-split input projection, the
    four norms, the attention in decode and prefill form, the TP dense MLP with its folded output scale, residuals)
    equals the reference ``MotifMTP`` in fp64 at T = 160 > window. Guards the wiring: concat order ``[hidden, embed]``,
    ``embed_norm`` on the embedding only, no ``hnorm``, which norm feeds what, both residuals, ``final_layernorm``."""
    cfg = host_cfg()
    args = ref_args(q_path_fp32=False)  # keep the q path in the module dtype (fp64)
    t = real_mtp_tensors() if weights_kind == "real" else random_mtp_tensors(args, seed=353)
    t64 = {k: v.double() for k, v in t.items()}
    ref = ref_mtp(args, t64, torch.float64)
    src = W.DictWeightSource(t64)
    T = 160
    g = torch.Generator().manual_seed(53)
    hn = torch.randn(T, cfg.hidden_size, generator=g, dtype=torch.float64)
    e = 3 * torch.randn(T, cfg.hidden_size, generator=g, dtype=torch.float64)  # RMS 3: embed_norm must act
    want = {k: v[0] for k, v in ref_steps(ref, hn[None], e[None], torch.arange(T)[None]).items()}
    for mode in ("decode", "prefill"):
        got = emulate_mtp(cfg, src, hn, e, mode)
        line = {k: stats(want[k], got[k])["rel_fro"] for k in ("h", "a", "o", "u", "out")}
        log(f"cpu fp64 MTP block {weights_kind} {mode}: rel_fro {', '.join(f'{k} {v:.2e}' for k, v in line.items())}")
        assert all(v < 2e-6 for v in line.values()), line  # the reference's norms / RoPE / PolyNorm run in fp32
    # negative control: the reverse concat order (DeepSeek-V3's [embed, hidden]) is far off
    w_in = t64[W.mtp_name("input_proj.weight")]
    swapped = W.DictWeightSource(
        {**t64, W.mtp_name("input_proj.weight"): torch.cat([w_in[:, 4096:], w_in[:, :4096]], 1)}
    )
    assert stats(want["out"], emulate_mtp(cfg, swapped, hn, e, "prefill")["out"])["rel_fro"] > 0.1


def test_cpu_mtp_next_tokens():
    """``mtp_next_tokens``: rows ``[s, e)`` take ``t_{p+1}``. Where the token at ``e`` is known (``e < n``: a
    generator-internal chunk end) row ``e - 1`` takes it, ``next_after_end`` may be omitted, and a different stand-in
    is refused (design §3.7.1: storing the argmax there would put a wrong MTP latent into a full, cacheable block). The
    last known row (``e == n``) requires the stand-in (the host argmax). Internal chunks concatenate to the one-shot
    tokens."""
    toks = torch.arange(100, 120)  # n = 20 known tokens
    assert MTPM.mtp_next_tokens(toks, 0, 20, 7).tolist() == list(range(101, 120)) + [7]  # last known row: stand-in
    assert MTPM.mtp_next_tokens(toks, 5, 9).tolist() == [106, 107, 108, 109]  # known next token
    assert MTPM.mtp_next_tokens(toks, 5, 9, int(toks[9])).tolist() == [106, 107, 108, 109]  # == tokens[9]: accepted
    assert MTPM.mtp_next_tokens(toks, 18, 19).dtype == torch.int32
    assert MTPM.mtp_next_tokens(toks.to(torch.int64), 19, 20, 3).tolist() == [3]
    with pytest.raises(ValueError, match="is known"):
        MTPM.mtp_next_tokens(toks, 5, 9, 7)  # a stand-in where the token at position 9 is known
    with pytest.raises(ValueError, match="is known"):
        MTPM.mtp_next_tokens(toks, 0, 19, 3)
    with pytest.raises(ValueError, match="next_after_end"):
        MTPM.mtp_next_tokens(toks, 0, 20)  # the last known row needs the stand-in
    for bad in ((0, 21), (5, 5), (-1, 3)):
        with pytest.raises(ValueError):
            MTPM.mtp_next_tokens(toks, *bad, 0)
    # a 300-token request prefilled as generator-internal chunks [0, 128), [128, 256), [256, 300): the same tokens as
    # one chunk, the argmax stand-in only on the last known row
    t = torch.randint(0, 219_520, (300,), generator=torch.Generator().manual_seed(3))
    whole = MTPM.mtp_next_tokens(t, 0, 300, 11)
    parts = [MTPM.mtp_next_tokens(t, 0, 128), MTPM.mtp_next_tokens(t, 128, 256), MTPM.mtp_next_tokens(t, 256, 300, 11)]
    assert torch.equal(torch.cat(parts), whole) and whole[-1] == 11 and torch.equal(whole[:-1], t[1:].to(torch.int32))


def test_cpu_fill_kv_prefill_chunk_checks(monkeypatch):
    """``MotifMTP.fill_kv_prefill(chunk=)`` hands the chunk itself to ``MotifAttention.fill_kv(chunk=)`` (whose
    ``_check_chunk_rows`` validates the rows and an sp1 chunk's RoPE rows), and refuses before any device op: ``hn``
    rows != ``chunk.bucket``, a chunk that is not a ``PrefillChunkInputs``, ``chunk=`` together with ``fill_pt=`` /
    ``rot=``, no table, no cache. ``fill_pt=`` / ``rot=`` reach the attention unchanged. The attention input is freed
    also when the attention raises (here: the real attention's check of an sp1 chunk without ``"plain"`` RoPE rows)."""
    from models.demos.motif3.tt.attention import PrefillChunkInputs

    fake = _FakeTTNN()
    monkeypatch.setattr(MTPM, "ttnn", fake)
    calls: List[tuple] = []

    class RecordingAttn:
        def fill_kv(self, x, **kw):
            calls.append(("attn.fill_kv", x, kw))

    def make(attn):
        m = object.__new__(MTPM.MotifMTP)
        m.attn = attn

        def block_input(hn, next_tokens, *, want_h, taps=None):  # stands in for the embedding + input projection
            calls.append(("input", hn, next_tokens))
            return None, _FT((1, 1, int(hn.shape[-2]), 4096))

        m._block_input_prefill = block_input
        return m

    def freed():
        return [c[1][0][1] for c in fake.calls if c[0] == "deallocate"]

    m = make(RecordingAttn())
    cache, fpt, rot = _FT((9, 1, 64, 576)), _FT((1, 2), "ROW_MAJOR"), {"plain": (_FT((1, 1, 128, 64)),) * 2}
    sp0 = PrefillChunkInputs("sp0", 0, 128, 100, fill_pt=fpt)
    sp1 = PrefillChunkInputs("sp1", 1024, 128, 1100, fill_pt=fpt, rot=rot)
    hn, nxt = _FT((1, 1, 128, 4096)), _FT((1, 128), "ROW_MAJOR")
    for ch in (sp0, sp1):
        calls.clear()
        m.fill_kv_prefill(hn, nxt, kv_cache=cache, chunk=ch)
        assert [c[0] for c in calls] == ["input", "attn.fill_kv"]
        assert calls[1][2] == {"chunk": ch, "kv_cache": cache, "taps": None}  # the chunk itself, not its tables
        assert calls[1][1].id in freed()  # the attention input
    calls.clear()
    m.fill_kv_prefill(hn, nxt, kv_cache=cache, fill_pt=fpt, rot=rot)
    assert calls[1][2] == {"fill_pt": fpt, "rot": rot, "kv_cache": cache, "taps": None}
    # refused before any device op
    from types import SimpleNamespace

    bad = [
        (dict(hn=_FT((1, 1, 256, 4096)), chunk=sp0), ValueError, "bucket"),  # 256 rows, a 128-row chunk
        (dict(hn=_FT((1, 1, 64, 4096)), chunk=sp1), ValueError, "bucket"),
        (dict(chunk=SimpleNamespace(fill_pt=fpt, rot=None, bucket=128)), TypeError, "PrefillChunkInputs"),
        (dict(chunk=sp0, fill_pt=fpt), ValueError, "not both"),
        (dict(chunk=sp1, rot=rot), ValueError, "not both"),
        (dict(), ValueError, "chunk="),
        (dict(chunk=sp0, kv_cache=None), ValueError, "kv_cache"),
    ]
    for kw, exc, msg in bad:
        calls.clear()
        fake.calls = []
        args = dict(hn=hn, kv_cache=cache)
        args.update(kw)
        with pytest.raises(exc, match=msg):
            m.fill_kv_prefill(args.pop("hn"), nxt, **args)
        assert calls == [] and fake.calls == [], (kw, calls)
    # the real attention check is reached (no longer bypassed): an sp1 chunk without the MTP layer's "plain" rows
    attn = object.__new__(MotifAttention)
    attn.cfg, attn.kind = host_cfg(), "plain"
    m = make(attn)
    calls.clear()
    fake.calls = []
    yarn_only = PrefillChunkInputs("sp1", 1024, 128, 1100, fill_pt=fpt, rot={"yarn": rot["plain"]})
    with pytest.raises(ValueError, match="'plain' RoPE rows"):
        m.fill_kv_prefill(hn, nxt, kv_cache=cache, chunk=yarn_only)
    assert [c[0] for c in calls] == ["input"] and len(freed()) == 1  # the input stage ran, its output was freed


@pytest.mark.parametrize("vocab_split", ["mesh", "tp"])
def test_cpu_lm_head_decode_ops_unchanged(vocab_split):
    """``MotifLMHead.forward_decode`` issues exactly the op sequence of the committed module (git ``HEAD``) for both
    vocab splits, with and without the device untilize (a recording fake ttnn: ops, operand lineage and shapes, kwarg
    names, deallocations). ``decode_logits(hn)`` is the same sequence after the stream-mean norm, except that it keeps
    ``hn`` (the speculative step feeds it to the MTP layer); ``consume=True`` frees it as ``forward_decode`` does."""
    from models.demos.motif3.tt import lm_head as cur

    fake = _FakeTTNN()
    old = _git_module("models/demos/motif3/tt/lm_head.py", "models.demos.motif3.tt._lm_head_head")
    for mod in (old, cur):
        mod.ttnn = fake
    try:
        for row_major in (False, True):
            seqs = []
            for mod in (old, cur):
                _FT._n = 0  # before the head's own fake weights: identical lineage ids for both modules
                h = _fake_head(mod, fake, vocab_split)
                fake.calls = []
                x = _FT((1, 4, 8, 4096))
                h.forward_decode(x, row_major=row_major)
                seqs.append(list(fake.calls))
            assert seqs[0] == seqs[1], f"{vocab_split} row_major={row_major}: forward_decode ops changed"
            n_norm = next(i for i, c in enumerate(seqs[1]) if c[0] == "to_memory_config" and c[1][1] == "'DRAM'") + 2
            _FT._n = 0
            h = _fake_head(cur, fake, vocab_split)
            fake.calls = []
            hn = _FT((1, 1, 8, 4096))
            h.decode_logits(hn, row_major=row_major)
            kept = list(fake.calls)
            assert ("deallocate", (("T", hn.id, hn.shape),), ()) not in kept, "decode_logits must keep hn"
            _FT._n = 0
            fake.calls = []
            h.decode_logits(_FT((1, 1, 8, 4096)), row_major=row_major, consume=True)
            assert [c[0] for c in fake.calls] == [c[0] for c in seqs[1][n_norm:]], "consume=True != forward_decode"
            log(
                f"lm_head {vocab_split} row_major={row_major}: {len(seqs[1])} ops, identical to HEAD; decode_logits "
                f"keeps hn ({len(kept)} ops)"
            )
    finally:
        import ttnn as real_ttnn

        for mod in (old, cur):
            mod.ttnn = real_ttnn


def test_cpu_embedding_rows_ops():
    """``MotifEmbedding.embed_rows`` (one gather of ``[1, T]`` + a view reshape to ``[1, 1, T, 4096]``) and
    ``embed_rows_from_device`` (``partition`` over DP of the lane-ordered ``[1, 1, 1, 32]`` argmax, then the same; the
    "tp" split's ``[1, 1, 1, 8]`` skips the partition); ``forward_decode`` / ``forward_prefill`` keep the committed op
    sequences. ``rows_tokens_host`` pads with ``pad_token_id``."""
    from models.demos.motif3.tt import embedding as cur

    fake = _FakeTTNN()
    old = _git_module("models/demos/motif3/tt/embedding.py", "models.demos.motif3.tt._embedding_head")
    cfg = host_cfg()

    def make(mod, shard_hidden=False):
        e = object.__new__(mod.MotifEmbedding)
        e.cfg, e.ccl, e.n_streams, e.hidden, e.lanes = cfg, _FakeCCL(fake), 4, 4096, 8
        e.shard_hidden, e.prefill_mode, e.memory_config = shard_hidden, "auto", "DRAM"
        e.weight = _FT((cfg.vocab_size, 512 if shard_hidden else 4096), "ROW_MAJOR")
        return e

    for mod in (old, cur):
        mod.ttnn = fake
    try:
        for shard_hidden in (False, True):
            for call in (lambda e: e.forward_decode(_FT((4, 8), "ROW_MAJOR")),
                         lambda e: e.forward_prefill(_FT((1, 256), "ROW_MAJOR")),
                         lambda e: e.forward_prefill(_FT((4, 2048), "ROW_MAJOR"))):  # fmt: skip
                seqs = []
                for mod in (old, cur):
                    _FT._n = 0
                    fake.calls = []
                    call(make(mod, shard_hidden))
                    seqs.append(list(fake.calls))
                assert seqs[0] == seqs[1]
            e = make(cur, shard_hidden)
            _FT._n = 0
            fake.calls = []
            out = e.embed_rows(_FT((1, 200), "ROW_MAJOR"))
            assert out.shape == (1, 1, 200, 4096) and [c[0] for c in fake.calls] == (
                ["embedding"] + (["ccl.all_gather.tp", "deallocate"] if shard_hidden else []) + ["reshape"]
            )
            fake.calls = []
            ids = _FT((1, 1, 1, 32), "ROW_MAJOR")
            out = e.embed_rows_from_device(ids)
            ops = [c[0] for c in fake.calls]
            assert out.shape == (1, 1, 8, 4096) and ops[0] == "ccl.partition.dp" and ops[-1] == "deallocate", ops
            assert ("deallocate", (("T", ids.id, ids.shape),), ()) not in fake.calls  # the argmax is not consumed
            fake.calls = []
            e.embed_rows_from_device(_FT((1, 1, 1, 8), "ROW_MAJOR"))
            assert "ccl.partition.dp" not in [c[0] for c in fake.calls] and fake.calls[-1][0] != "deallocate"
            with pytest.raises(ValueError):
                e.embed_rows_from_device(_FT((1, 1, 1, 16), "ROW_MAJOR"))
            with pytest.raises(ValueError):
                e.embed_rows(_FT((4, 16), "ROW_MAJOR"))
    finally:
        import ttnn as real_ttnn

        for mod in (old, cur):
            mod.ttnn = real_ttnn
    row = cur.prefill_token_row(torch.tensor([5, 6, 7]), 32, cfg, n_copies=1)
    assert row.shape == (1, 32) and row[0, :3].tolist() == [5, 6, 7] and int(row[0, 3:].unique()) == cfg.pad_token_id


def _converter():
    path = PROJECT_ROOT / "scripts" / "convert_weights.py"
    name = "motif3_scripts_convert_weights"  # the name tests/test_weight_cache.py and stream_weights.py use
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _json_doc(out: str) -> dict:
    """The ``--status --json`` document in captured stdout (other prints may precede it)."""
    start = out.find('{\n "cache_root"')
    assert start >= 0, out[:400]
    return json.loads(out[start:])


def test_cpu_converter_mtp_part(tmp_path, capsys):
    """``scripts/convert_weights.py`` part ``L53`` (kind ``mtp``): kind / spec / estimate (the measured 420,183,552
    bytes of the 21 files), no option variants, the BF16 source check (19 tensors; a source without them is refused),
    the part status on a synthetic directory, the ``--status --json --mtp`` document, and the main kinds' estimates
    unchanged. The two script copies are identical."""
    cw = _converter()
    pkg_copy = REPO_ROOT / "models" / "demos" / "motif3" / "scripts" / "convert_weights.py"
    assert pkg_copy.read_bytes() == (PROJECT_ROOT / "scripts" / "convert_weights.py").read_bytes()
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8), tt_cache_root=tmp_path, num_layers=53)
    assert cw.mtp_part(cfg) == 53 and cw.part_kind(cfg, 53) == "mtp" and cw.part_tag(53) == "L53"
    assert cw.part_kind(cfg, 52) == "moe" and cw.part_kind(cfg, 1) == "dense" and cw.part_kind(cfg, None) == "global"
    assert cw.part_spec(cfg, 53) == cfg.mtp_layer_spec() and cw.part_spec(cfg, 4) == cfg.layer(4)
    assert cw.EST_PART_BYTES == {"global": 3_607_113_408, "dense": 355_205_440, "moe": 6_655_598_208}
    assert cw.estimate_part_bytes("mtp") == 420_183_552 and cw.ConvertOptions().variants_for("mtp") == {}
    assert cw.MTP_PREFIX == W.mtp_name() and "models/demos/motif3/tt/mtp.py" in cw.CODE_FINGERPRINT_FILES
    assert cw.module_of("mtp.v1.input_proj") == "mtp" and cw.module_of("mtp.v1.final_layernorm.weight") == "mtp"
    assert cw.mtp_parts(cfg, True) == [53] and cw.mtp_parts(cfg, False) == []
    assert cw.missing_source_tensors(W.HFWeightLoader(), cfg, 53) == []
    assert cw.missing_source_tensors(W.DictWeightSource({}), cfg, 53) == [
        "model.mtp_layers.0.* (no tensors in the index)"
    ]
    # a partial index: the tensors it does not list are missing too (the part would fail at load time otherwise)
    full = {n: torch.zeros(1) for n in W.mtp_tensor_names()}
    assert cw.missing_source_tensors(W.DictWeightSource(full), cfg, 53) == []
    partial = {k: v for k, v in full.items() if "input_proj" not in k and "mlp.act_fn" not in k}
    assert (
        cw.missing_source_tensors(W.DictWeightSource(partial), cfg, 53)
        == [W.mtp_name("input_proj.weight"), W.mtp_name("mlp.act_fn.bias"), W.mtp_name("mlp.act_fn.weight")]
        == sorted(set(full) - set(partial))
    )

    class NoMTP:
        num_nextn_predict_layers = 0

    assert cw.mtp_part(NoMTP()) is None and cw.mtp_parts(NoMTP(), True) == []
    # part status on a synthetic L53 directory
    opts = cw.ConvertOptions()
    st = cw.part_status(cfg, 53, opts)
    assert not st.complete and st.kind == "mtp" and st.need_bytes == 420_183_552
    d = W.layer_cache_marker(cfg, 53).parent
    d.mkdir(parents=True)
    (d / "mtp.v1.input_proj__tp0_dtype_BFLOAT16_layout_TILE.tensorbin").write_bytes(b"x" * 10)
    W.mark_layer_cached(cfg, 53, ["mtp.v1.input_proj__tp0_dtype_BFLOAT16_layout_TILE.tensorbin"])
    cw.write_json_atomic(d / cw.STATS_NAME, {"files": {"mtp.v1.input_proj__tp0_dtype_BFLOAT16_layout_TILE.tensorbin": {
        "bytes": 10, "sha256": "0" * 64}}, "variants": {}, "verify": {"ok": True, "files_hashed": 1}})  # fmt: skip
    st = cw.part_status(cfg, 53, opts)
    assert st.complete and st.verified and st.as_dict()["part"] == "L53" and st.as_dict()["kind"] == "mtp"
    # the --status document (no device): --mtp alone lists only L53
    rc = cw.main(["--status", "--json", "--mtp", "--cache-root", str(tmp_path)])
    doc = _json_doc(capsys.readouterr().out)
    assert rc == 0 and [p["part"] for p in doc["parts"]] == ["L53"] and doc["parts"][0]["complete"]
    rc = cw.main(["--status", "--json", "--layers", "0", "--globals", "--cache-root", str(tmp_path)])
    doc = _json_doc(capsys.readouterr().out)
    assert [p["part"] for p in doc["parts"]] == ["global", "L00"]  # explicit parts: no MTP unless --mtp
    rc = cw.main(["--status", "--json", "--cache-root", str(tmp_path)])  # the default run: everything + the MTP part
    doc = _json_doc(capsys.readouterr().out)
    assert [p["part"] for p in doc["parts"]][0] == "global" and [p["part"] for p in doc["parts"]][-1] == "L53"
    assert len(doc["parts"]) == 55
    # the serving cache (informational: converted on this host by work package 3)
    real = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8), num_layers=53)
    rst = cw.part_status(real, 53, opts)
    log(
        f"serving cache part L53: complete {rst.complete} verified {rst.verified} files {rst.files} bytes {rst.bytes} "
        f"({rst.reason})"
    )
    if rst.complete:
        assert rst.files == 21 and rst.bytes == 420_183_552 and rst.verified is True


def test_cpu_mtp_goldens():
    """The reference goldens (built on first use) and the CPU acceptance they imply: the bf16 reference reproduces the
    estimate of ``spec_mtp.md`` §1.2 (0.760 over all 2965 rows, 0.835 on the 1970 on-policy rows) and the fp32
    reference is within a point of it."""
    gold = load_goldens()
    R = int(gold["target"].numel())
    assert R == 2965 and int(gold["on_policy"].sum()) == 1970
    assert gold["hn"].shape == (R, 4096) and gold["out"].shape == (R, 4096) and gold["hn"].dtype == torch.bfloat16
    a16, a32 = acceptance(gold["m16"], gold), acceptance(gold["m32"], gold)
    agree = float((gold["m16"] == gold["m32"]).float().mean())
    nt = non_tie_rows(gold)
    log(
        f"CPU acceptance: bf16 reference all {a16['all']:.4f} on-policy {a16['on_policy']:.4f} (no tie "
        f"{a16['on_policy_no_tie']:.4f}); fp32 reference all {a32['all']:.4f} on-policy {a32['on_policy']:.4f}; "
        f"bf16 vs fp32 MTP argmax agree {agree:.4f}; non-tie rows {int(nt.sum())}/{R}"
    )
    # spec_mtp.md ran the bf16 reference with 6 threads; bf16 CPU GEMMs round per thread split, so a few rows flip
    assert abs(a16["all"] - ALPHA_CPU["all"]) < 0.005 and abs(a16["on_policy"] - ALPHA_CPU["on_policy"]) < 0.005
    assert abs(a32["all"] - a16["all"]) < 0.01 and abs(a32["on_policy"] - a16["on_policy"]) < 0.01
    for name in gold["prompts"]:
        o, n = gold["rows"][name]
        ids = gold["ids"][name]
        assert n == ids.numel() - 1 and torch.equal(gold["next_ids"][o : o + n], ids[1:])


# ======================================================================================================================
# device helpers
# ======================================================================================================================
class _RaisingSource:
    """A weight source that must never be read: the modules must load every tensor from the TT cache."""

    def _fail(self, *a, **k):
        raise AssertionError(f"the BF16 source was read: {a}")

    get = get_rows = shape = available = has = keys = _fail

    def __contains__(self, name):
        self._fail(name)


def build_mtp_modules(mesh_device, cfg, ccl, rope):
    """The globals (embedding, LM head "mesh") and the MTP layer from the serving TT cache with a raising source and
    without any write (``model.read_only_cache``): proves part ``L53`` is complete. Skips when not converted."""
    from models.demos.motif3.tt.embedding import MotifEmbedding
    from models.demos.motif3.tt.lm_head import MotifLMHead
    from models.demos.motif3.tt.model import layer_cache_complete, read_only_cache

    for part in (None, cfg.mtp_layer_idx):
        if not layer_cache_complete(cfg, part):
            pytest.skip(f"TT-cache part {'global' if part is None else f'L{part}'} is not converted "
                        f"(scripts/convert_weights.py --globals --mtp)")  # fmt: skip
    src = _RaisingSource()
    t0 = time.time()
    with read_only_cache() as misses:
        embed = MotifEmbedding(mesh_device, cfg, source=src, ccl=ccl, cache=True)
        head = MotifLMHead(mesh_device, cfg, source=src, ccl=ccl, cache=True)
        mtp = MTPM.MotifMTP(mesh_device, cfg, source=src, ccl=ccl, rope=rope, embed=embed, head=head, cache=True)
    assert misses == [], f"tensors missing from the TT cache: {misses}"
    log(f"globals + MTP layer (L53) loaded from the TT cache alone in {time.time() - t0:.1f} s")
    return embed, head, mtp


def _stale_cache(mesh_device, cfg, pool: int, seed: int):
    """Replicated MTP cache ``[pool, 1, block, 576]`` whose every slot holds a stale row (unit-RMS latent + random
    k_pe), so an unwritten or wrongly written row shows."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.randn(pool * cfg.kv_block_size, cfg.kv_latent_dim, generator=g)
    rows[:, : cfg.kv_lora_rank] = _rmsnorm(rows[:, : cfg.kv_lora_rank], 1e-5)
    host = _host_quant(rows.reshape(pool, 1, cfg.kv_block_size, -1), cfg.dtypes.kv_cache)
    return _upload_cache(mesh_device, host, cfg.dtypes.kv_cache), host


def _rows_tt(mesh_device, x: torch.Tensor, rows: int):
    """``x [n, 4096]`` -> replicated ``[1, 1, rows, 4096]`` bf16 (zero rows after ``n``: finite padding)."""
    pad = torch.zeros(rows, x.shape[-1])
    pad[: x.shape[0]] = x.float()
    return _replicated(mesh_device, pad[None, None], ttnn.bfloat16)


def _page_table(pool: int, n: int, seed: int) -> torch.Tensor:
    return (torch.randperm(pool - 1, generator=torch.Generator().manual_seed(seed))[:n] + 1).to(torch.int32)


def _tokens_tt(mesh_device, tok32: torch.Tensor):
    """Lane-ordered ``[32]`` ids -> ``[1, 1, 1, 32]`` uint32 ROW_MAJOR on every chip (``argmax_decode``'s output)."""
    return ttnn.from_torch(tok32.reshape(1, 1, 1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))  # fmt: skip


def _host_tokens(tok32: torch.Tensor, mesh_device):
    return ttnn.from_torch(tok32.reshape(1, 1, 1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))  # fmt: skip


def _all_equal_chips(t, mesh_device, chips: Optional[Sequence[int]] = None) -> bool:
    """The readbacks of ``chips`` (default all 32; device-tensor index = row-major mesh coordinate) are identical."""
    devs = ttnn.get_device_tensors(t)
    idx = list(range(len(devs))) if chips is None else list(chips)
    first = ttnn.to_torch(devs[idx[0]])
    return all(torch.equal(first, ttnn.to_torch(devs[i])) for i in idx[1:])


def _decode_lanes(gold: Dict[str, Any], cfg) -> tuple:
    """Lane ``l`` (DP row ``r = l // 8``) decodes prompt ``DECODE_PROMPTS[r]`` at ``p = DECODE_POS[r][l % 8]``.
    Returns ``(positions [32], golden rows of the positions [32], per lane the golden rows of positions 0 .. HIST-1)``;
    ``p + 1`` (the trace replay's and a draft's position) is inside the history and the prompt."""
    lane_pos, lane_rows, hist_rows = [], [], []
    for lane in range(cfg.max_batch):
        r, j = divmod(lane, cfg.lanes_per_row)
        o, n = gold["rows"][DECODE_PROMPTS[r]]
        p = DECODE_POS[r][j]
        assert p + 1 < HIST <= n
        lane_pos.append(p)
        lane_rows.append(o + p)
        hist_rows.append(torch.arange(o, o + HIST))
    return lane_pos, lane_rows, hist_rows


def _fill_histories(mesh_device, cfg, rope, embed, mtp, gold, cache, pt_lanes, hist_rows, n_single: int = 16):
    """KV-only fills of every lane's positions ``0 .. HIST-1`` into ``cache``: lanes ``[0, n_single)`` one
    :meth:`MotifMTP.fill_kv_prefill` each (the serving path), the other lanes in ONE call (their rows concatenated with
    their fill tables and per-row RoPE positions, 2 row chunks)."""

    def ds(rows):
        return gold["hn"][rows].float(), gold["next_ids"][rows]

    for lane in range(n_single):
        hn, nxt = ds(hist_rows[lane])
        hn_tt, tok_tt = _rows_tt(mesh_device, hn, HIST), embed.rows_tokens_device(nxt, HIST)
        fill_tt = _replicated(mesh_device, pt_lanes[lane][None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        mtp.fill_kv_prefill(hn_tt, tok_tt, kv_cache=cache, fill_pt=fill_tt)
        _free([hn_tt, tok_tt, fill_tt])
    B = len(hist_rows)
    if n_single >= B:
        return
    packed = torch.cat(hist_rows[n_single:])
    hn, nxt = ds(packed)
    hn_tt, tok_tt = _rows_tt(mesh_device, hn, packed.numel()), embed.rows_tokens_device(nxt, packed.numel())
    fill_tt = _replicated(mesh_device, pt_lanes[n_single:].reshape(1, -1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    idx_tt = rope.chunk_rot_idxs_device(torch.arange(HIST).repeat(B - n_single))
    rot = rope.chunk_rope_tables(idx_tt, kinds=("plain",))
    mtp.row_chunk = packed.numel() // 2  # 2 row chunks
    try:
        mtp.fill_kv_prefill(hn_tt, tok_tt, kv_cache=cache, fill_pt=fill_tt, rot=rot)
    finally:
        mtp.row_chunk = cfg.prefill_row_chunk
    _free([hn_tt, tok_tt, fill_tt, idx_tt] + [t for cs in rot.values() for t in cs])


def _slot(pt_row: torch.Tensor, q: int, block: int):
    """``(block id, row in block)`` of position ``q`` in a page-table row."""
    return int(pt_row[q // block]), q % block


def _poisoned(
    hist: torch.Tensor, stale: torch.Tensor, pt_lanes: torch.Tensor, slots: Dict[int, Sequence[int]], block: int
) -> torch.Tensor:
    """``hist`` with lane ``l``'s cache rows at the positions ``slots[l]`` replaced by the stale rows of ``stale``: a
    decode step must write those slots itself (the history fill already wrote them with the decode's own inputs, so a
    missing or misplaced write would otherwise go unnoticed)."""
    out = hist.clone()
    for lane, qs in slots.items():
        for q in qs:
            b, i = _slot(pt_lanes[lane], q, block)
            out[b, 0, i] = stale[b, 0, i]
    return out


def _decode_writes(cache, mesh_device, cfg, before, step, mode: str, copies=None):
    """Every chip's copy of ``cache`` after one decode step vs the host model of ``step``'s KV writes under ``mode``
    (``kv_write.apply_kv_writes_host`` from ``before``: each active lane's slot, on the chips the mode writes -- its
    own DP row for ``row`` / ``row_split``, all 32 chips for ``all`` / ``all_split`` -- and nothing else). The written
    rows are read from each lane's own DP row (chip ``(dp, 0)``). ``copies``: an earlier readback of the same state
    (``None`` = read now). Returns ``(rows [B, 576], {(dp, tp): elements that differ from the model}, {(dp, tp):
    copy})``."""
    copies = KW.cache_copies(cache, mesh_device, cfg) if copies is None else copies
    rows = torch.zeros(step.lanes, cfg.kv_latent_dim)
    for lane, b, i in step.write_slots(cfg.kv_block_size):
        rows[lane] = copies[(lane // cfg.lanes_per_row, 0)][b, 0, i]
    expected = KW.apply_kv_writes_host(
        before, step, mode, rows, block_size=cfg.kv_block_size, lanes_per_row=cfg.lanes_per_row
    )
    bad = {k: int((v != expected[k[0]]).sum()) for k, v in copies.items() if not torch.equal(v, expected[k[0]])}
    return rows, bad, copies


def _latent_stats(ref, gold, gam: torch.Tensor, rank: int, got: torch.Tensor, gold_rows, positions):
    """Cache rows ``got [n, 576]`` vs the fp32 reference latents of the golden rows at ``positions``: ``(n * gamma,
    k_pe)`` stats and the worst per-row PCC of either."""
    n_ref, kpe_ref = ref_latents(
        ref.self_attn, gold["a"][torch.as_tensor(gold_rows).long()], torch.as_tensor(positions).long()
    )
    s_n, s_k = stats(n_ref * gam, got[:, :rank] * gam), stats(kpe_ref, got[:, rank:])
    worst = min(min(pcc(n_ref[i] * gam, got[i, :rank] * gam), pcc(kpe_ref[i], got[i, rank:])) for i in range(len(got)))
    return s_n, s_k, worst


def _mtp_decode_step(mesh_device, cfg, rope, mtp, head, cache, positions, x32, tok32, pt, kvw=None, step=None):
    """One eager MTP decode step: lane ``l`` at ``positions[l]`` (``-1`` = inactive) with the main model's hidden
    ``x32[l]``, the main argmax ``tok32[l]`` and the page-table row ``pt[l]``. ``kvw``: a ``kv_write.DecodeKVWrite``
    that writes ``step`` (``write_step``; its ``cur_pos`` / ``page_table`` feed the attention, as WP5 wires every
    layer); ``None`` = the draft-1 write. Returns ``(m [32], {"attn_out", "out"}: [32, 4096], replicas identical)``;
    frees everything it allocated."""
    from models.demos.motif3.tt.ccl import replicas_identical

    positions = torch.as_tensor(positions).to(torch.int32)
    if kvw is not None:
        kvw.write_step(step)
        cur, ptt = kvw.cur_pos, kvw.page_table
    else:
        cur = _rows(mesh_device, cfg, positions, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        ptt = _rows(mesh_device, cfg, torch.where((positions >= 0)[:, None], pt, 0).to(torch.int32), ttnn.int32,
                    ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
    x_tt, tok_tt = _lane_x(mesh_device, cfg, x32), _tokens_tt(mesh_device, tok32)
    rot_idx = rope.rot_idxs_device(positions)
    rot = MotifAttention.decode_rope_tables(rope, rot_idx)
    act = MotifAttention.active_mask_from_cur_pos(cur, cfg.lanes_per_row)
    taps: Dict[str, Any] = {}
    m_tt = mtp.forward_decode(x_tt, tok_tt, rot=rot, cur_pos=cur, page_table=ptt, kv_cache=cache, active=act,
                              kv_write=kvw, taps=taps)  # fmt: skip
    m = head.tokens_to_host(m_tt)
    res = {k: _per_lane_out(cfg, mesh_device, taps[k]) for k in ("attn_out", "out")}
    rep = replicas_identical(taps["out"], mesh_device, "tp", cfg.axes) and _all_equal_chips(m_tt, mesh_device)
    for k in list(taps):
        _free(taps.pop(k))
    _free([m_tt, x_tt, tok_tt, rot_idx, act] + [t for cs in rot.values() for t in cs])
    if kvw is None:
        _free([cur, ptt])
    return m, res, rep


# ======================================================================================================================
# device tests
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mtp_g14(mesh_device, device_params):
    """Gate G14 with the real MTP weights from the TT cache (part L53) and realistic inputs (C2 golden hidden states,
    the checkpoint's embedding rows of the next tokens):

    1. KV-only prefill fill (:meth:`MotifMTP.fill_kv_prefill`) at C = 128 (sp0, a shared block -1), 1024 (the serving
       form ``chunk=``: the ``PrefillChunkInputs`` of a planned resumed row [1024, 1924), an sp1 chunk with the
       offset-RoPE rows, whose cached prefix and pure-padding block are skipped) and 4096 (sp0; again with 4 row
       chunks): cache rows vs the reference ``c_kv / gamma`` and ``k_pe``; skipped, prefix and other blocks untouched.
    2. One 32-lane decode step past the window (positions 129 .. 510; DP row r on prompt r) over histories written by
       fill_kv_prefill (16 lanes one call each, 16 lanes in one packed call with 2 row chunks), with each lane's slots
       p and p + 1 then poisoned with stale rows (the fill wrote them with the decode's own inputs): GDLA output per
       lane, the MLP on its own input, the MTP output per lane, the MTP argmax vs the fp32 reference; the decode's KV
       write on all 32 chips vs the host model (p rewritten on the lane's own DP row, nothing else anywhere) and the
       written latents vs the reference.
    3. The decode step traced, captured over the histories with every p + 1 poisoned: the replay at p + 1 writes p + 1
       (all-chip host model, written latents vs the reference) and an eager step from the same cache state gives
       bitwise the same tokens, hidden states and cache copies on all 32 chips; traced time."""
    from models.demos.motif3.tt.attention import PrefillChunkInputs, chunk_host_tables
    from models.demos.motif3.tt.ccl import replicas_identical

    gold = load_goldens()
    cfg, ccl, rope = _setup(mesh_device, "mtp g14")
    embed, head, mtp = build_mtp_modules(mesh_device, cfg, ccl, rope)
    args = ref_args()
    t32 = real_mtp_tensors(torch.float32)
    ref = ref_mtp(args, t32, torch.float32)
    gam = t32[W.mtp_name("self_attn.kv_norm.weight")].float()
    block, rank, kvdt = cfg.kv_block_size, cfg.kv_lora_rank, cfg.dtypes.kv_cache
    R = int(gold["target"].numel())
    failures: List[str] = []

    def dataset(rows: torch.Tensor):
        return gold["hn"][rows].float(), gold["next_ids"][rows], gold["a"][rows]

    # ---- (0) the LM head's split decode path (spec trace) == forward_decode, bitwise on every chip -----------------
    from models.demos.motif3.tt.ccl import device_tensors_to_torch
    from models.demos.motif3.tt.rope import shard_lanes

    x4h = torch.randn(
        cfg.dp, cfg.n_streams, cfg.lanes_per_row, cfg.hidden_size, generator=torch.Generator().manual_seed(4)
    )
    x4h = (x4h * 4).to(torch.bfloat16).float()
    x4 = shard_lanes(x4h, cfg, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    ref_logits = head.forward_decode(x4, row_major=True)
    hn4 = head.stream_mean_norm(x4)
    split_logits = head.decode_logits(hn4, row_major=True)
    kept = hn4.is_allocated()
    eq_head = torch.equal(
        device_tensors_to_torch(ref_logits, mesh_device), device_tensors_to_torch(split_logits, mesh_device)
    )
    log(f"LM head: stream_mean_norm + decode_logits == forward_decode bitwise on 32 chips {eq_head}; hn kept {kept}")
    if not (eq_head and kept):
        failures.append(f"LM head split path: bitwise {eq_head}, hn kept {kept}")
    _free([x4, ref_logits, hn4, split_logits])

    # ---- (1) KV-only prefill fill -----------------------------------------------------------------------------------
    for C, start, real_end, chunk_rows in ((128, 0, 128, None), (1024, 1024, 1924, None), (4096, 0, 4096, None),
                                          (4096, 0, 4096, 1024)):  # fmt: skip
        rows = torch.arange(C) % R
        hn, nxt, a_ref = dataset(rows)
        nb = C // block
        inp = None
        if start:  # the serving form: the PrefillChunkInputs of the planned request row [start, real_end)
            plan = cfg.plan_prefill_row(start, real_end)
            ch0 = plan.chunks[0]
            assert len(plan.chunks) == 1 and (ch0.start, ch0.bucket, ch0.end, ch0.path) == (start, C, real_end, "sp1")
            n_req = -(-real_end // block)  # the request's blocks: positions 0 .. real_end - 1
            pool = 1 + n_req + 2
            pt_req = _page_table(pool, n_req, seed=C + start)
            host = chunk_host_tables(cfg, plan, ch0, pt_req)
            fill = host.fill[0].clone()  # chunk-local block j -> block id, -1 = skip
            pt = torch.where(fill >= 0, fill, 0)
            never = [int(b) for b in pt_req[: start // block]]  # the cached prefix (read-only)
        else:
            pool = 1 + nb + 2
            pt = _page_table(pool, nb, seed=C + start)
            fill = pt.clone()
            if C == 128:
                fill[0] = -1  # a shared (cached) full block below w0: never written
            never = [int(pt[j]) for j in range(nb) if int(fill[j]) < 0]
        skip = [j for j in range(nb) if int(fill[j]) < 0]
        cache, stale = _stale_cache(mesh_device, cfg, pool, seed=C)
        hn_tt = _rows_tt(mesh_device, hn, C)
        tok_tt = embed.rows_tokens_device(nxt, C)
        old_chunk = mtp.row_chunk
        if chunk_rows is not None:
            mtp.row_chunk = chunk_rows
        try:
            if start:
                inp = PrefillChunkInputs.upload(mesh_device, cfg, rope, host)
                mtp.fill_kv_prefill(hn_tt, tok_tt, kv_cache=cache, chunk=inp)
            else:
                fill_tt = _replicated(mesh_device, fill[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                mtp.fill_kv_prefill(hn_tt, tok_tt, kv_cache=cache, fill_pt=fill_tt)
                _free(fill_tt)
        finally:
            mtp.row_chunk = old_chunk
        ch = _dev(cache)
        positions = torch.arange(start, start + C)
        written = torch.tensor([i for i in range(C) if (i // block) not in skip and start + i < real_end])
        # fill-table entry j holds the chunk's rows [64 j, 64 j + 64): gather by chunk-local row, compare at positions
        n_ref, kpe_ref = ref_latents(ref.self_attn, a_ref[written], positions[written])
        got = _gather_cache(ch, pt, written, block)
        s_n, s_k = stats(n_ref * gam, got[:, :rank] * gam), stats(kpe_ref, got[:, rank:])
        targets = {int(b) for b in pt[fill >= 0]}
        untouched = all(torch.equal(ch[b], stale[b]) for b in never)
        clean = all(torch.equal(ch[b], stale[b]) for b in range(pool) if b not in targets)
        same_chips = _all_equal_chips(cache, mesh_device, chips=(0, 13, 31))  # fill writes every chip
        tag = f"C={C} start={start}" + (" chunk=PrefillChunkInputs" if start else "")
        tag += f" row_chunk={chunk_rows}" if chunk_rows else ""
        log(
            f"fill_kv_prefill {tag}: cache n*gamma vs ref {fmt(s_n)}; k_pe {fmt(s_k)}; {len(written)} rows written, "
            f"{'cached-prefix' if start else 'skipped'} blocks ({len(never)}) untouched {untouched}, every other "
            f"block untouched {clean}, 32 chips identical {same_chips}"
        )
        if not (_good(s_n, LAT_PCC_MIN) and _good(s_k, LAT_PCC_MIN) and untouched and clean and same_chips):
            failures.append(f"fill {tag}: n {fmt(s_n)} k_pe {fmt(s_k)} untouched {untouched} clean {clean}")
        if C == 4096:
            if chunk_rows is None:
                unchunked = ch.clone()
            else:
                eq = torch.equal(unchunked, ch)
                s_c = stats(unchunked, ch)
                log(f"fill_kv_prefill C=4096: 4 row chunks vs one pass: bitwise {eq} ({fmt(s_c)})")
                if not (eq or s_c["pcc"] >= 0.99999):
                    failures.append(f"chunked fill differs: {fmt(s_c)}")
        if inp is not None:
            inp.free()
        _free([cache, hn_tt, tok_tt])

    # ---- (1b) eager cost of the KV-only fill per chunk bucket (it adds to every prefill chunk's TTFT) ---------------
    fill_ms = {}
    for C in (128, 1024, 8192):
        rows = torch.arange(C) % R
        hn, nxt, _ = dataset(rows)
        cache, _ = _stale_cache(mesh_device, cfg, 1 + C // block, seed=7)
        hn_tt, tok_tt = _rows_tt(mesh_device, hn, C), embed.rows_tokens_device(nxt, C)
        fill_tt = _replicated(mesh_device, torch.arange(1, 1 + C // block, dtype=torch.int32)[None], ttnn.int32,
                              ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        times = []
        for _ in range(3):  # the first call compiles
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            mtp.fill_kv_prefill(hn_tt, tok_tt, kv_cache=cache, fill_pt=fill_tt)
            ttnn.synchronize_device(mesh_device)
            times.append((time.perf_counter() - t0) * 1e3)
        fill_ms[C] = min(times[1:])
        _free([cache, hn_tt, tok_tt, fill_tt])
    log("fill_kv_prefill eager wall time (synchronized, min of 2 after the compile): "
        + ", ".join(f"C={C} {v:.1f} ms" for C, v in fill_ms.items()))  # fmt: skip

    # ---- (2) 32-lane decode over fill_kv_prefill histories ----------------------------------------------------------
    B, per = cfg.max_batch, HIST // block
    pool = 1 + B * per + 3
    pt_lanes = _page_table(pool, B * per, seed=99).reshape(B, per)
    cache, stale = _stale_cache(mesh_device, cfg, pool, seed=98)
    lane_pos, lane_rows, hist_rows = _decode_lanes(gold, cfg)
    t0 = time.time()
    _fill_histories(mesh_device, cfg, rope, embed, mtp, gold, cache, pt_lanes, hist_rows)
    ttnn.synchronize_device(mesh_device)
    log(
        f"decode histories: 16 x fill_kv_prefill(C={HIST}) + one packed fill of 16 x {HIST} rows in "
        f"{time.time() - t0:.2f} s"
    )
    hist = _dev(cache)
    hist_chips = _all_equal_chips(cache, mesh_device, chips=(0, 13, 31))
    lat_worst = 1.0
    for lane in (0, 9, 18, 27):
        hp = torch.arange(lane_pos[lane])
        _, _, a_h = dataset(hist_rows[lane][: lane_pos[lane]])
        n_ref, kpe_ref = ref_latents(ref.self_attn, a_h, hp)
        got = _gather_cache(hist, pt_lanes[lane], hp, block)
        s_n, s_k = stats(n_ref * gam, got[:, :rank] * gam), stats(kpe_ref, got[:, rank:])
        lat_worst = min(lat_worst, s_n["pcc"], s_k["pcc"])
        if not (_good(s_n, LAT_PCC_MIN) and _good(s_k, LAT_PCC_MIN)):
            failures.append(f"history lane {lane}: n {fmt(s_n)} k_pe {fmt(s_k)}")
    # The fill wrote every lane's decode slot p and p + 1 (HIST = 512 > 510 + 1) from the decode's own inputs: poison
    # both with stale rows, so only the decode's own write can make p right, and a write to p + 1 shows.
    poisoned = _poisoned(hist, stale, pt_lanes, {l: (lane_pos[l], lane_pos[l] + 1) for l in range(B)}, block)
    _free(cache)
    cache = _upload_cache(mesh_device, poisoned, kvdt)
    exact_up = torch.equal(_dev(cache), poisoned)
    rows_t = torch.tensor(lane_rows)
    # negative control: before the step, the slots p hold rows far from the reference (the write check can fail)
    pre = torch.stack([poisoned[_slot(pt_lanes[l], lane_pos[l], block)[0], 0, lane_pos[l] % block] for l in range(B)])
    pre_n, pre_k, _ = _latent_stats(ref, gold, gam, rank, pre, rows_t, lane_pos)
    log(
        f"histories: 32 chips identical {hist_chips}; p / p + 1 of every lane poisoned, re-upload exact {exact_up}; "
        f"poisoned p vs ref (negative control) n*gamma pcc {pre_n['pcc']:.4f} k_pe pcc {pre_k['pcc']:.4f}"
    )
    if not (hist_chips and exact_up and pre_n["pcc"] < 0.5 and pre_k["pcc"] < 0.5):
        failures.append(f"histories: chips identical {hist_chips}, poisoned re-upload exact {exact_up}, poisoned p vs "
                        f"ref {pre_n['pcc']:.4f} / {pre_k['pcc']:.4f} (must be far)")  # fmt: skip
    hn32, tok32, a32 = dataset(rows_t)
    d = _decode_step_inputs(mesh_device, cfg, rope, lane_pos, hn32, pt_lanes)
    tok_tt = _tokens_tt(mesh_device, tok32)
    taps: Dict[str, Any] = {}
    m_tt = mtp.forward_decode(d["x"], tok_tt, rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache,
                              active=d["act"], taps=taps)  # fmt: skip
    m = head.tokens_to_host(m_tt)
    got = {k: _per_lane_out(cfg, mesh_device, taps[k]) for k in ("e", "attn_out", "ffn_in", "ffn_out", "out", "h")}
    exact_e = torch.equal(got["e"], gold["e_next"][rows_t].float())  # embed(t_{p+1}) rows: the checkpoint's, bit-exact
    if not exact_e:
        failures.append("embed_rows_from_device rows differ from the checkpoint embedding rows")
    s_attn = stats(gold["o"][rows_t], got["attn_out"])
    s_out = stats(gold["out"][rows_t], got["out"])
    with torch.no_grad():
        mlp_ref = ref.mlp(got["ffn_in"])  # the reference MLP on the TT's own MLP input
    s_mlp = stats(mlp_ref, got["ffn_out"])
    lane_attn = [pcc(gold["o"][rows_t][l], got["attn_out"][l]) for l in range(B)]
    lane_out = [pcc(gold["out"][rows_t][l], got["out"][l]) for l in range(B)]
    lane_mlp = [pcc(mlp_ref[l], got["ffn_out"][l]) for l in range(B)]
    nt = non_tie_rows(gold)[rows_t]
    agree = m == gold["m32"][rows_t]
    rep_ok = replicas_identical(taps["out"], mesh_device, "tp", cfg.axes) and _all_equal_chips(m_tt, mesh_device)
    log(
        f"MTP decode (32 lanes at 129..510, real weights): GDLA {fmt(s_attn)} worst lane {min(lane_attn):.6f}; MLP "
        f"on its own input {fmt(s_mlp)} worst lane {min(lane_mlp):.6f}; output {fmt(s_out)} worst lane "
        f"{min(lane_out):.6f}; "
        f"argmax == fp32 ref {int(agree.sum())}/32 ({int(agree[nt].sum())}/{int(nt.sum())} without a bf16 tie); "
        f"replicas identical {rep_ok}; history latents worst pcc {lat_worst:.6f}; embed rows bit-exact {exact_e}"
    )
    if not (_good(s_attn, ATTN_PCC_MIN) and min(lane_attn) >= ATTN_PCC_MIN):
        failures.append(f"decode GDLA: {fmt(s_attn)} worst lane {min(lane_attn)}")
    if not (_good(s_mlp, MLP_PCC_MIN) and min(lane_mlp) >= MLP_PCC_MIN):
        failures.append(f"decode MLP: {fmt(s_mlp)} worst lane {min(lane_mlp)}")
    if not (_good(s_out, OUT_PCC_MIN) and min(lane_out) >= OUT_PCC_MIN):
        failures.append(f"decode output: {fmt(s_out)} worst lane {min(lane_out)}")
    if float(agree[nt].float().mean()) < ARGMAX_AGREE_MIN or not rep_ok:
        failures.append(f"decode argmax agree {agree.tolist()} (non-tie {nt.tolist()}), replicas {rep_ok}")
    # the decode's KV write (draft 1 = kv_write "row"): each lane's p rewritten on its own DP row's 8 chips; p + 1, the
    # other rows' copies of p and every other cache row unchanged on all 32 chips; the written rows = the ref latents
    step_a = KW.KVWriteStep.ordinary(torch.tensor(lane_pos, dtype=torch.int32), pt_lanes)
    w_rows, w_bad, _ = _decode_writes(cache, mesh_device, cfg, poisoned, step_a, "row")
    landed = 0
    for l in range(B):
        b, i = _slot(pt_lanes[l], lane_pos[l], block)
        landed += int(not torch.equal(w_rows[l], poisoned[b, 0, i]))
    w_n, w_k, w_lane = _latent_stats(ref, gold, gam, rank, w_rows, rows_t, lane_pos)
    log(
        f"MTP decode KV write (p, p + 1 poisoned): p rewritten on the own DP row {landed}/32; all 32 chips == the "
        f"host model ('row': p on the own row only, p + 1 and every other row unchanged) {not w_bad} {w_bad}; "
        f"written latents vs ref: n*gamma {fmt(w_n)}, k_pe {fmt(w_k)}, worst lane {w_lane:.6f}"
    )
    if w_bad or landed != B or not (_good(w_n, LAT_PCC_MIN) and _good(w_k, LAT_PCC_MIN) and w_lane >= LAT_ROW_MIN):
        failures.append(f"decode KV write: landed {landed}/32, chips off the model {w_bad}, n {fmt(w_n)} k_pe "
                        f"{fmt(w_k)} worst lane {w_lane}")  # fmt: skip
    for k in list(taps):
        _free(taps.pop(k))
    _free(m_tt)

    # ---- (3) trace: replay with new inputs == eager, traced time ---------------------------------------------------
    rot_idx_p = rope.rot_idxs_device(torch.tensor(lane_pos))
    x_p, cur_p, pt_p, tok_p = d["x"], d["cur"], d["pt"], tok_tt

    def step(kv):
        rot_s = MotifAttention.decode_rope_tables(rope, rot_idx_p, kinds=("plain",))
        act_s = MotifAttention.active_mask_from_cur_pos(cur_p, cfg.lanes_per_row)
        m_s, out_s = mtp.forward_decode(x_p, tok_p, rot=rot_s, cur_pos=cur_p, page_table=pt_p, kv_cache=kv,
                                        active=act_s, return_hidden=True)  # fmt: skip
        _free([act_s] + [t for cs in rot_s.values() for t in cs])
        return m_s, out_s

    _free(list(step(cache)))  # eager: compiles every program of the step
    if os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):  # mock devices have no dispatch: trace capture cannot run
        log("mock cluster: trace section skipped")
        _free([rot_idx_p, cache])
        _free_step(d)
        _free(tok_tt)
        mtp.deallocate()
        assert not failures, "\n".join(failures)
        return
    # S0 = the histories (p as the fill wrote it) with every lane's p + 1 poisoned: the replay at p + 1 must write it
    pos_b = [p + 1 for p in lane_pos]
    rows_b = torch.tensor(lane_rows) + 1
    s0 = _poisoned(hist, stale, pt_lanes, {l: (pos_b[l],) for l in range(B)}, block)
    pre = torch.stack([s0[_slot(pt_lanes[l], pos_b[l], block)[0], 0, pos_b[l] % block] for l in range(B)])
    pre_n, pre_k, _ = _latent_stats(ref, gold, gam, rank, pre, rows_b, pos_b)
    if not (pre_n["pcc"] < 0.5 and pre_k["pcc"] < 0.5):  # negative control, as in (2)
        failures.append(f"trace: poisoned p + 1 vs ref {pre_n['pcc']:.4f} / {pre_k['pcc']:.4f} (must be far)")
    _free(cache)
    cache = _upload_cache(mesh_device, s0, kvdt)  # before the capture: the trace binds this buffer
    with _Capture(mesh_device) as cap:
        m_tr, out_tr = step(cache)
    try:
        hn_b, tok_b, _ = dataset(rows_b)
        hn_host = shard_lanes(
            hn_b.reshape(cfg.dp, 1, 8, -1), cfg, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(hn_host, x_p)
        ttnn.copy_host_to_device_tensor(shard_lanes(torch.tensor(pos_b, dtype=torch.int32), cfg, mesh_device), cur_p)
        ttnn.copy_host_to_device_tensor(rope.rot_idxs_host(torch.tensor(pos_b)), rot_idx_p)
        ttnn.copy_host_to_device_tensor(_host_tokens(tok_b, mesh_device), tok_p)
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
        m_trace, out_trace = head.tokens_to_host(m_tr), _per_lane_out(cfg, mesh_device, out_tr)
        step_b = KW.KVWriteStep.ordinary(torch.tensor(pos_b, dtype=torch.int32), pt_lanes)
        t_rows, t_bad, t_copies = _decode_writes(cache, mesh_device, cfg, s0, step_b, "row")
        t_n, t_k, t_lane = _latent_stats(ref, gold, gam, rank, t_rows, rows_b, pos_b)
        # the eager step from the same state, on a fresh copy of S0 (allocated after the capture, freed before the
        # next replay): its own write of p + 1, not the replay's, is what the cache comparison sees
        cache_e = _upload_cache(mesh_device, s0, kvdt)
        m_e, out_e = step(cache_e)
        m_eager, out_eager = head.tokens_to_host(m_e), _per_lane_out(cfg, mesh_device, out_e)
        e_copies = KW.cache_copies(cache_e, mesh_device, cfg)
        _free([m_e, out_e, cache_e])
        eq_cache = all(torch.equal(t_copies[k], e_copies[k]) for k in t_copies)
        del t_copies, e_copies
        eq = torch.equal(m_trace, m_eager) and torch.equal(out_trace, out_eager) and eq_cache
        s_b = stats(gold["out"][rows_b], out_trace)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            times.append((time.perf_counter() - t0) * 1e3)
        log(
            f"MTP decode traced: replay (positions +1, new inputs) == eager from the same cache bitwise {eq} (tokens, "
            f"hidden, all 32 cache copies {eq_cache}); replay output vs ref {fmt(s_b)}; replay's KV write (p + 1 "
            f"poisoned: n*gamma pcc {pre_n['pcc']:.4f} before): all 32 chips == the host model {not t_bad} {t_bad}, "
            f"latents n*gamma {fmt(t_n)}, k_pe "
            f"{fmt(t_k)}, worst lane {t_lane:.6f}; traced step {min(times):.3f} ms (min of 5, incl. the RoPE / "
            f"active-mask gathers)"
        )
        if not (eq and _good(s_b, OUT_PCC_MIN)):
            failures.append(f"trace replay: bitwise {eq} (cache {eq_cache}), output {fmt(s_b)}")
        if t_bad or not (_good(t_n, LAT_PCC_MIN) and _good(t_k, LAT_PCC_MIN) and t_lane >= LAT_ROW_MIN):
            failures.append(f"trace replay KV write: chips off the model {t_bad}, n {fmt(t_n)} k_pe {fmt(t_k)} "
                            f"worst lane {t_lane}")  # fmt: skip
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    # traced cost per decode step (slope of 4 vs 8 steps per trace; RoPE / mask tensors: the step's shared ones)
    rot_f = MotifAttention.decode_rope_tables(rope, rot_idx_p, kinds=("plain",))
    act_f = MotifAttention.active_mask_from_cur_pos(cur_p, cfg.lanes_per_row)
    _, out_f = mtp.forward_decode(x_p, tok_p, rot=rot_f, cur_pos=cur_p, page_table=pt_p, kv_cache=cache, active=act_f,
                                  return_hidden=True)  # fmt: skip

    def mtp_step():
        return mtp.forward_decode(x_p, tok_p, rot=rot_f, cur_pos=cur_p, page_table=pt_p, kv_cache=cache, active=act_f)

    def head_part():
        lg = head.decode_logits(out_f)
        ids = head.argmax_decode(lg)
        ttnn.deallocate(lg)
        return ids

    t_mtp = _traced_us(mesh_device, mtp_step, n=8, reps=5)
    t_head = _traced_us(mesh_device, head_part, n=8, reps=5)
    log(
        f"MTP decode step traced: {t_mtp['slope_us']:.1f} us per step (raw {t_mtp['raw_us']:.1f}), of which the shared "
        f"head (ag_dp_rows + project + argmax) {t_head['slope_us']:.1f} us (raw {t_head['raw_us']:.1f})"
    )
    _free([m_tr, out_tr, rot_idx_p, cache, act_f, out_f] + [t for cs in rot_f.values() for t in cs])
    _free_step(d)
    _free(tok_tt)
    mtp.deallocate()
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mtp_decode_kv_writes(mesh_device, device_params):
    """G14, decode side: the MTP layer's KV write in every serving configuration and its lane independence (the WP3
    review probe, folded in). Histories as in :func:`test_mtp_g14` (KV-only fills of positions 0 .. 511); every lane's
    decode slot p and p + 1 is then poisoned with a stale row, so only the step's own write makes it right. Every case
    checks all 32 chips' cache copies against the ``kv_write`` host model (the slots the mode writes, on the chips it
    writes, nothing else; :func:`_decode_writes`) and compares outputs, argmax and written rows bitwise with case A or
    with the fp32 reference:

    A. the draft-1 write (``kv_write=None`` = mode ``row``), 32 lanes at p: GDLA / output / argmax and the written
       latents vs the reference.
    B. odd lanes inactive (position -1): the active lanes bitwise == A; the inactive lanes write nothing; every
       output row finite.
    C. every lane moved to the next DP row (same token, hidden state, position and page table): bitwise == A.
    D. ``kv_write`` mode ``all`` (KV-R, an ordinary step): every chip holds every lane's p (A's rows, bitwise);
       outputs bitwise == A.
    E. ``all_split`` packed verify: owners 0..15 at p (call A), their drafts on partner lanes 16..31 -- other DP rows
       -- at p + 1 with the owner's page table (call B): owners bitwise == A; partner GDLA / output / argmax and the
       partner-written p + 1 vs the reference; p and p + 1 on every chip.
    E2. an ordinary step at p + 1 on the owners' own lanes (``all``, the cache after E): bitwise == E's partner rows
       (outputs, argmax, written rows): a draft row's result does not depend on the lane or DP row that computes it
       (features design §3.8.2, review R5)."""
    gold = load_goldens()
    cfg, ccl, rope = _setup(mesh_device, "mtp kv writes")
    embed, head, mtp = build_mtp_modules(mesh_device, cfg, ccl, rope)
    t32 = real_mtp_tensors(torch.float32)
    ref = ref_mtp(ref_args(), t32, torch.float32)
    gam = t32[W.mtp_name("self_attn.kv_norm.weight")].float()
    block, rank, kvdt = cfg.kv_block_size, cfg.kv_lora_rank, cfg.dtypes.kv_cache
    B, per, L = cfg.max_batch, HIST // block, cfg.lanes_per_row
    h = B // 2
    pool = 1 + B * per + 3
    pt_lanes = _page_table(pool, B * per, seed=99).reshape(B, per)
    cache, stale = _stale_cache(mesh_device, cfg, pool, seed=98)
    lane_pos, lane_rows, hist_rows = _decode_lanes(gold, cfg)
    t0 = time.time()
    _fill_histories(mesh_device, cfg, rope, embed, mtp, gold, cache, pt_lanes, hist_rows)
    hist = _dev(cache)
    _free(cache)
    poisoned = _poisoned(hist, stale, pt_lanes, {l: (lane_pos[l], lane_pos[l] + 1) for l in range(B)}, block)
    pos_t, rows_p = torch.tensor(lane_pos, dtype=torch.int32), torch.tensor(lane_rows)
    x_a, tok_a = gold["hn"][rows_p].float(), gold["next_ids"][rows_p]
    nt = non_tie_rows(gold)
    failures: List[str] = []

    def check(tag: str, ok: bool, detail: str) -> None:
        log(f"{tag}: {'ok' if ok else 'FAIL'}; {detail}")
        if not ok:
            failures.append(f"{tag}: {detail}")

    def run(c, positions, x32, tok32, pt, kvw=None, step=None):
        return _mtp_decode_step(mesh_device, cfg, rope, mtp, head, c, positions, x32, tok32, pt, kvw=kvw, step=step)

    def eq_rows(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], ia, ib) -> bool:
        return all(torch.equal(a[k][ia], b[k][ib]) for k in a)

    def argmax_ok(m: torch.Tensor, gr: torch.Tensor) -> tuple:
        agree = m == gold["m32"][gr]
        sel = nt[gr]
        return float(agree[sel].float().mean()) >= ARGMAX_AGREE_MIN, f"{int(agree.sum())}/{len(gr)}"

    def outputs_vs_ref(out: Dict[str, torch.Tensor], gr: torch.Tensor, lanes) -> tuple:
        s_attn, s_out = stats(gold["o"][gr], out["attn_out"][lanes]), stats(gold["out"][gr], out["out"][lanes])
        return _good(s_attn, ATTN_PCC_MIN) and _good(s_out, OUT_PCC_MIN), f"GDLA {fmt(s_attn)}; out {fmt(s_out)}"

    def latents_vs_ref(got: torch.Tensor, gr, positions) -> tuple:
        s_n, s_k, worst = _latent_stats(ref, gold, gam, rank, got, gr, positions)
        ok = _good(s_n, LAT_PCC_MIN) and _good(s_k, LAT_PCC_MIN) and worst >= LAT_ROW_MIN
        return ok, f"n*gamma {s_n['pcc']:.6f} k_pe {s_k['pcc']:.6f} worst row {worst:.6f}"

    c = _upload_cache(mesh_device, poisoned, kvdt)
    exact = torch.equal(_dev(c), poisoned)
    log(f"histories filled and p / p + 1 of every lane poisoned in {time.time() - t0:.2f} s; re-upload exact {exact}")
    if not exact:
        failures.append("the poisoned history does not re-upload exactly")

    # ---- A: the draft-1 write, 32 lanes at p ---------------------------------------------------------------------
    t0 = time.time()
    m_a, out_a, rep_a = run(c, pos_t, x_a, tok_a, pt_lanes)
    st_a = KW.KVWriteStep.ordinary(pos_t, pt_lanes)
    rows_a, bad, copies_a = _decode_writes(c, mesh_device, cfg, poisoned, st_a, "row")
    ok_o, d_o = outputs_vs_ref(out_a, rows_p, slice(None))
    ok_m, d_m = argmax_ok(m_a, rows_p)
    ok_l, d_l = latents_vs_ref(rows_a, rows_p, pos_t)
    # negative controls on the same readback: the KV-R model ("all": every lane's p on every chip) misses the other
    # 24 lanes' p on every chip, the "nothing written" model the own 8 lanes' p on every chip
    _, bad_all, _ = _decode_writes(c, mesh_device, cfg, poisoned, st_a, "all", copies=copies_a)
    _, bad_none, _ = _decode_writes(c, mesh_device, cfg, poisoned, KW.KVWriteStep.inactive(per), "row", copies=copies_a)
    neg = len(bad_all) == B and len(bad_none) == B
    del copies_a
    ok = ok_o and ok_m and ok_l and rep_a and not bad and neg
    check("A draft-1 write (kv_write None = 'row'), 32 lanes at p", ok,
          f"{d_o}; argmax == fp32 ref {d_m}; replicas {rep_a}; written p vs ref {d_l}; all 32 chips == the host model "
          f"{not bad} {bad}; negative controls: chips off the 'all' model {len(bad_all)} (want {B}), off the "
          f"'nothing written' model {len(bad_none)} (want {B}) ({time.time() - t0:.1f} s incl. the 32-chip "
          f"readback)")  # fmt: skip
    _free(c)

    # ---- B: odd lanes inactive -----------------------------------------------------------------------------------
    c = _upload_cache(mesh_device, poisoned, kvdt)
    pos_b = torch.where(torch.arange(B) % 2 == 1, torch.full_like(pos_t, -1), pos_t)
    m_b, out_b, _ = run(c, pos_b, x_a, tok_a, pt_lanes)
    rows_b, bad, _ = _decode_writes(c, mesh_device, cfg, poisoned, KW.KVWriteStep.ordinary(pos_b, pt_lanes), "row")
    act = torch.arange(0, B, 2)
    eq = eq_rows(out_b, out_a, act, act) and torch.equal(m_b[act], m_a[act]) and torch.equal(rows_b[act], rows_a[act])
    fin = all(bool(torch.isfinite(v).all()) for v in out_b.values())
    check("B odd lanes inactive", eq and fin and not bad,
          f"active lanes bitwise == A (outputs, argmax, written rows) {eq}; every output row finite {fin}; all 32 "
          f"chips == the host model (inactive lanes write nothing) {not bad} {bad}")  # fmt: skip
    _free(c)

    # ---- C: every lane relocated to the next DP row --------------------------------------------------------------
    c = _upload_cache(mesh_device, poisoned, kvdt)
    perm = torch.tensor([(l + L) % B for l in range(B)])  # original lane l runs on lane perm[l]
    inv = torch.argsort(perm)  # lane j runs original lane inv[j]
    st_c = KW.KVWriteStep.ordinary(pos_t[inv], pt_lanes[inv])
    m_c, out_c, _ = run(c, pos_t[inv], x_a[inv], tok_a[inv], pt_lanes[inv])
    rows_c, bad, _ = _decode_writes(c, mesh_device, cfg, poisoned, st_c, "row")
    eq = eq_rows(out_c, out_a, perm, slice(None)) and torch.equal(m_c[perm], m_a) and torch.equal(rows_c[perm], rows_a)
    diff = {k: float((out_c[k][perm] - out_a[k]).abs().max()) for k in out_c}
    check("C every lane on the next DP row", eq and not bad,
          f"bitwise == A (outputs, argmax, written rows) {eq} (max |diff| {diff}); all 32 chips == the host model "
          f"{not bad} {bad}")  # fmt: skip
    _free(c)

    # ---- D: kv_write "all" (KV-R), ordinary step -----------------------------------------------------------------
    c = _upload_cache(mesh_device, poisoned, kvdt)
    kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=per, mode="all")
    st_d = KW.KVWriteStep.ordinary(pos_t, pt_lanes)
    m_d, out_d, _ = run(c, pos_t, x_a, tok_a, pt_lanes, kvw=kvw, step=st_d)
    rows_d, bad, _ = _decode_writes(c, mesh_device, cfg, poisoned, st_d, "all")
    eq = eq_rows(out_d, out_a, slice(None), slice(None)) and torch.equal(m_d, m_a) and torch.equal(rows_d, rows_a)
    check("D kv_write 'all' (KV-R)", eq and not bad,
          f"outputs, argmax and written rows bitwise == A {eq}; all 32 chips == the host model (every lane's p on "
          f"every chip) {not bad} {bad}")  # fmt: skip
    kvw.deallocate()
    _free(c)

    # ---- E: kv_write "all_split", packed verify with the partners on other DP rows -------------------------------
    c = _upload_cache(mesh_device, poisoned, kvdt)
    kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=per, mode="all_split")
    pos_own = torch.where(torch.arange(B) < h, pos_t, torch.full_like(pos_t, -1))
    st_e = KW.KVWriteStep.packed_verify(pos_own, pt_lanes, {o: o + h for o in range(h)})
    gr_e = rows_p.clone()
    gr_e[h:] = rows_p[:h] + 1  # a partner runs its owner's draft: the hidden state / token of the owner's p + 1
    x_e, tok_e = gold["hn"][gr_e].float(), gold["next_ids"][gr_e]
    m_e, out_e, _ = run(c, st_e.positions, x_e, tok_e, st_e.page_table, kvw=kvw, step=st_e)
    rows_e, bad, copies_e = _decode_writes(c, mesh_device, cfg, poisoned, st_e, "all_split")
    own = torch.arange(h)
    eq_own = eq_rows(out_e, out_a, own, own) and torch.equal(m_e[:h], m_a[:h]) and torch.equal(rows_e[:h], rows_a[:h])
    ok_o, d_o = outputs_vs_ref(out_e, gr_e[h:], slice(h, B))
    ok_m, d_m = argmax_ok(m_e[h:], gr_e[h:])
    ok_l, d_l = latents_vs_ref(rows_e[h:], gr_e[h:], st_e.positions[h:])
    check("E kv_write 'all_split' packed verify (partners 16..31 on DP rows 2-3 at p + 1)",
          eq_own and ok_o and ok_m and ok_l and not bad,
          f"owners bitwise == A {eq_own}; partners {d_o}; argmax == fp32 ref {d_m}; partner-written p + 1 vs ref "
          f"{d_l}; all 32 chips == the host model (p by call A, p + 1 by call B, on every chip) {not bad} "
          f"{bad}")  # fmt: skip
    after_e = copies_e[(0, 0)].clone()
    del copies_e
    kvw.deallocate()

    # ---- E2: ordinary step at p + 1 on the owners' own lanes (the cache after E) ---------------------------------
    kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=per, mode="all")
    pos_e2 = torch.where(torch.arange(B) < h, pos_t + 1, torch.full_like(pos_t, -1))
    x_e2, tok_e2 = torch.zeros_like(x_e), torch.zeros_like(tok_e)
    x_e2[:h], tok_e2[:h] = x_e[h:], tok_e[h:]
    st_e2 = KW.KVWriteStep.ordinary(pos_e2, pt_lanes)
    m_e2, out_e2, _ = run(c, pos_e2, x_e2, tok_e2, pt_lanes, kvw=kvw, step=st_e2)
    rows_e2, bad, _ = _decode_writes(c, mesh_device, cfg, after_e, st_e2, "all")
    eq = (eq_rows(out_e2, out_e, own, slice(h, B)) and torch.equal(m_e2[:h], m_e[h:])
          and torch.equal(rows_e2[:h], rows_e[h:]))  # fmt: skip
    diff = {k: float((out_e2[k][:h] - out_e[k][h:]).abs().max()) for k in out_e2}
    check("E2 ordinary step at p + 1 on the owners' lanes", eq and not bad,
          f"== E's partner rows bitwise (outputs, argmax, written p + 1) {eq} (max |diff| {diff}); all 32 chips == "
          f"the host model {not bad} {bad}")  # fmt: skip
    kvw.deallocate()
    _free(c)
    mtp.deallocate()
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mtp_acceptance_c2(mesh_device, device_params):
    """The TT MTP layer on the C2 golden hidden states, teacher-forced over all 2965 rows (the full block in prefill
    form,
    :meth:`MotifMTP.forward_prefill`, one call per prompt; argmax through the shared LM head): output vs the fp32
    reference per row (G14 output bar), MTP argmax vs the fp32 reference (>= 99 % of the non-tie rows), and the
    acceptance rate vs the CPU estimate (``spec_mtp.md`` §1.2) within 2 points."""
    gold = load_goldens()
    cfg, ccl, rope = _setup(mesh_device, "mtp acceptance")
    embed, head, mtp = build_mtp_modules(mesh_device, cfg, ccl, rope)
    R = int(gold["target"].numel())
    m_tt = torch.zeros(R, dtype=torch.long)
    out_pccs = torch.zeros(R)
    failures: List[str] = []
    agg = []
    t0 = time.time()
    for name in gold["prompts"]:
        o, n = gold["rows"][name]
        bucket = max(128, 1 << (n - 1).bit_length())
        hn_tt = _rows_tt(mesh_device, gold["hn"][o : o + n], bucket)
        tok_tt = embed.rows_tokens_device(gold["next_ids"][o : o + n], bucket)
        out = mtp.forward_prefill(hn_tt, tok_tt)
        m_tt[o : o + n] = mtp.rows_argmax(out, n)
        got = _dev(out)[0, 0, :n]
        ref = gold["out"][o : o + n]
        s = stats(ref, got)
        out_pccs[o : o + n] = torch.tensor([pcc(ref[i], got[i]) for i in range(n)])
        agg.append((ref, got))
        a = acceptance(m_tt, gold, rows=torch.arange(R).ge(o) & torch.arange(R).lt(o + n))
        log(
            f"{name}: {n} rows (bucket {bucket}): output vs fp32 ref {fmt(s)}; alpha {a['all']:.4f} on-policy "
            f"{a['on_policy']:.4f}"
        )
        if not _good(s, OUT_PCC_MIN):
            failures.append(f"{name} output: {fmt(s)}")
        _free([out, hn_tt, tok_tt])
    s_all = stats(torch.cat([r for r, _ in agg]), torch.cat([g for _, g in agg]))
    nt = non_tie_rows(gold)
    agree_all = float((m_tt == gold["m32"]).float().mean())
    agree_nt = float((m_tt == gold["m32"])[nt].float().mean())
    a_tt, a16, a32 = acceptance(m_tt, gold), acceptance(gold["m16"], gold), acceptance(gold["m32"], gold)
    log(
        f"TT MTP over {R} C2 rows in {time.time() - t0:.1f} s: output {fmt(s_all)}, per-row pcc min "
        f"{float(out_pccs.min()):.5f} median {float(out_pccs.median()):.6f}; argmax == fp32 ref {agree_all:.4f} "
        f"({agree_nt:.4f} of {int(nt.sum())} non-tie rows)"
    )
    log(
        f"acceptance (teacher-forced on the golden hidden states): TT all {a_tt['all']:.4f} on-policy "
        f"{a_tt['on_policy']:.4f} (no tie {a_tt['on_policy_no_tie']:.4f}); CPU bf16 ref {a16['all']:.4f} / "
        f"{a16['on_policy']:.4f}; CPU fp32 ref {a32['all']:.4f} / {a32['on_policy']:.4f}"
    )
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    (TEST_DIR / "acceptance_tt.json").write_text(json.dumps(
        {"tt": a_tt, "cpu_bf16": a16, "cpu_fp32": a32, "argmax_agree_fp32": agree_all, "argmax_agree_non_tie": agree_nt,
         "output_pcc": s_all["pcc"], "row_pcc_min": float(out_pccs.min())}, indent=1))  # fmt: skip
    if not _good(s_all, OUT_PCC_MIN):
        failures.append(f"aggregate output {fmt(s_all)}")
    if agree_nt < ARGMAX_AGREE_MIN:
        failures.append(f"MTP argmax agreement on non-tie rows {agree_nt:.4f} < {ARGMAX_AGREE_MIN}")
    for k in ("all", "on_policy"):
        if abs(a_tt[k] - ALPHA_CPU[k]) > ALPHA_TOL:
            failures.append(f"acceptance {k} {a_tt[k]:.4f} vs CPU {ALPHA_CPU[k]} (tol {ALPHA_TOL})")
    mtp.deallocate()
    assert not failures, "\n".join(failures)
