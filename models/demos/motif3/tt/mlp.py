# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dense PolyNorm MLP (layers 0-1) and the MoE shared expert (layers 2-52), TP8 over the 8 chips of a DP row.

Design §2.3.5 / §2.3.6 / §2.3.7 decode step 10, §3.2; WAVE_A_REVIEW §5.5 MLP-1..3. HF semantics (``MotifMLP``,
``modeling_motif.py:473-501``; reference ``modules.MLP``)::

    out = down( bf16(poly(gate(x)) * up(x)) * 0.5 )          poly: scalar PolyNorm, bias NOT clamped

Per chip (``tp`` = this chip's TP index; ``n = I / 8``: dense 12288 -> 1536, shared 1280 -> 160)::

    gu = x @ W_gu[tp]            [T, 4096] @ [4096, 2n] -> [T, 2n] fp32 (column-parallel; gate | up, mlp_gate_up_for_chip)
    h  = PolyNorm_TP(gu[:, :n], gu[:, n:])                   moments sum g^2/g^4/g^6 all-reduced over TP (1 small CCL)
    y  = h @ W_dn[tp]            [T, n] @ [n, 4096] -> [T, 4096] bf16 partial (row-parallel, x0.5 folded, mlp_down_for_chip)
    y  = all_reduce(y, "tp")                                 (skipped with all_reduce=False: the MoE adds its routed partial)

Weights bfp8 (``cfg.dtypes.dense_mlp`` / ``shared_expert``), compute roles ``dense_mlp`` / ``shared`` (HiFi4, fp32 dest
acc, INFRA-3) for the matmuls and ``polynorm`` for the PolyNorm reductions. Decode matmuls use explicit
1D-multicast program configs (M = one tile; :func:`build_decode_program_configs` = :func:`decode_matmul_pc` on the grids of
:data:`DECODE_MATMUL_GRIDS`, from the traced sweep ``tests/unit/test_mlp.py::test_mlp_device_variants``; 1.2-4x faster
than the auto configs). A tuned grid that does not fit ``cfg.compute_grid`` (a differently harvested chip) or the
matmul shape falls back to ttnn's auto config (logged; ``PolyNormMLP.decode_pc_fallbacks``). These builders are a
local helper: moving them into ``model_config`` (``cfg.*_pc()``, README §5 / §10.1) is a requested shared change.
Prefill uses ttnn's auto config, in row chunks of ``cfg.prefill_row_chunk`` (8192) rows for long prompts (design §3.3).

Decode defaults: intermediates in L1; the 8 lanes are padded to one full tile of logical rows inside the module
(``pad_decode_rows``; avoids implicit-padding fills in the reductions and CCLs) and sliced back at the end; PolyNorm
moments all-reduced exactly with ``ar="ag_sum"`` (see tt/polynorm.py). Traced per call on this Galaxy (random weights
at real dims, bfp8; ``test_mlp_device_variants``): dense 198.5 us (166.0 us with ``all_reduce=False``), shared expert
132.5 us (101.4 us partial); the first version (8 logical rows, ``ttnn.all_reduce`` moments) took 215.7 / 181.6 and
147.6 / 114.7 us; DRAM intermediates +12 / +3 us.

The row padding writes into the caller's tile padding: ``ttnn.pad`` within a tile is ``fill_implicit_tile_padding``
(in place: ``FillPadDeviceOperation`` returns its input) + a view (``pad.cpp:478-480``), so ``forward_decode`` zero-fills
rows T..31 of ``x``'s last tile row in its own buffer (DRAM; no copy). Logical rows are never modified. Do not pass a
T-row *view* of a larger buffer whose rows T..31 hold data (they would be zeroed); every ttnn op output owns its
buffer, so normal callers are unaffected.

Statistics modes (``stats``): ``"tp"`` (default, design §2.3.6: gate | up column-parallel, 3-moment all-reduce over
TP) or ``"replicated_gate"`` (shared expert option): every chip also holds the full gate ``[4096, I]`` and takes the
moments locally (exact, no moments CCL); its TP gate shard comes from ``ccl.partition`` of the full gate, up and down
stay TP-sharded. Same accuracy (identical PCC / max-abs in the tests: same bfp8 blocks, exact moments either way);
measured 136.4 / 102.0 us (no gain over "tp") for +4.9 MB of bfp8 weights per shared expert per chip, so "tp" stays
the default.

Shapes at the module boundary (README CONVENTIONS §3): decode ``x [1, 1, T, 4096]`` bf16 TILE DRAM (T = 8 lanes of
this DP row, any T <= 32 works), replicated in the row -> ``[1, 1, T, 4096]`` bf16 DRAM, replicated in the row after
the all-reduce (bitwise identical on the 8 TP chips). Prefill ``x [1, 1, S, 4096]`` (S a multiple of 32; the
dense layers get the full replicated prompt, the shared expert may get any row slice, e.g. the MoE's ``S/4`` rows of
this DP row, ``MotifMoE.dp_slice``) -> ``[1, 1, S, 4096]``. Inputs are never consumed and their logical rows are never
modified (decode: see the tile-padding note above).

MoE integration (shared expert): ``part = shared.forward_decode(f, all_reduce=False)`` (this chip's TP partial,
bf16) -> ``moe.forward_decode(f, add_partial=part)`` so one ``ar_tp`` closes routed + shared; prefill:
``moe.forward_prefill(f, add_partial=shared.forward_prefill(moe.dp_slice(f), all_reduce=False))``.

Cache names (``weights.as_tensor``): dense ``mlp.gate_up``, ``mlp.down``; shared ``moe.shared.gate_up``,
``moe.shared.down`` (TP-sharded on the last / row dim: ``__tp3`` / ``__tp2``); ``stats="replicated_gate"`` adds
``moe.shared.gate_full`` (``__rep``) and ``moe.shared.up`` (``__tp3``) instead of ``gate_up``. PolyNorm constants:
``mlp.polynorm.{D,E,b}`` / ``moe.shared.polynorm.{D,E,b}`` (``__rep``, 3 tiny tensors + ``b`` in bf16). Every source
read is lazy (weights, ``act_fn.{weight,bias}``): with all files cached, the module is built from the TT cache alone
(``test_mlp_device_random_weights`` checks it with a source that raises on any read).

Output scale: ``polynorm.polynorm_output_scale(cfg, l)`` (0.5, folded into ``W_down``; exact for a power of two);
``polynorm.check_polynorm_semantics(cfg)`` rejects ``polynorm_sigmoid_weight=False`` once the config parses it.

Fused shared-expert PolyNorm (B5, ``shared_polynorm="fused"``, ``MOTIF3_SHARED_POLYNORM``; docs/OPTIMIZATION_PLAN.md
§3.3): the decode PolyNorm of the shared expert runs as ``tt/kernels/shared_polynorm.py`` (a one-core moments kernel
on the gate_up output, the release's moments all-gather, a one-core apply kernel: 3 programs instead of 19). The
kernels issue the release's LLK operations in the release's order, so the output is bitwise the composite's
(:func:`resolve_shared_polynorm` admits it only with the release's decode settings: fp32 PolyNorm, ``stats="tp"``, the
default ``moments`` / ``horner`` / ``ar`` knobs). Prefill and the dense MLPs keep the composite.

Trace safety: ``forward_decode`` has fixed shapes, per-layer constant slices, no host round trips and frees its
intermediates in a fixed order; every program is compiled by the first (eager) call.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn

from . import weights as W
from .ccl import MotifCCL
from .model_config import SHARED_POLYNORM_MODES, TILE, MotifTTConfig
from .polynorm import (
    POLYNORM_MODES,
    PolyNormCoefficients,
    ScalarPolyNormConsts,
    check_polynorm_semantics,
    polynorm_output_scale,
    polynorm_tp,
)

MLP_KINDS = ("dense", "shared")
MLP_STATS = ("tp", "replicated_gate")
# the polynorm_tp knobs of the release decode path; B5's fused kernels reproduce exactly these (resolve_shared_polynorm)
RELEASE_PN_KW = dict(moments="sum", horner="mac", ar="ag_sum")


def resolve_shared_polynorm(shared_polynorm: Optional[str], cfg, *, kind: str, stats: str, decode_polynorm: str,
                            pn_kw: dict, n_local: int) -> str:
    """B5: the decode PolyNorm a :class:`PolyNormMLP` runs. ``shared_polynorm`` (explicit) or ``cfg.shared_polynorm``
    (``MOTIF3_SHARED_POLYNORM``; ``None`` / absent = "composite") must be in :data:`SHARED_POLYNORM_MODES`. "fused"
    needs the shared expert (``kind="shared"``), ``stats="tp"``, the fp32 decode PolyNorm, the release's
    ``moments`` / ``horner`` / ``ar`` knobs (:data:`RELEASE_PN_KW`: the kernels reproduce that path bit for bit) and
    a tile-aligned local width the one-core kernels hold: an explicit request raises otherwise, the config default
    falls back to "composite" (the dense MLPs, diagnostic variants)."""
    explicit = shared_polynorm is not None
    mode = str(shared_polynorm if explicit else (getattr(cfg, "shared_polynorm", None) or "composite"))
    if mode not in SHARED_POLYNORM_MODES:
        raise ValueError(f"shared_polynorm must be one of {SHARED_POLYNORM_MODES}, got {mode!r}")
    if mode == "fused":
        from .kernels.shared_polynorm import MAX_TILES

        why = None
        if kind != "shared":
            why = f"the shared expert (kind={kind!r})"
        elif stats != "tp":
            why = f"stats='tp' (got {stats!r})"
        elif decode_polynorm != "fp32":
            why = f"decode_polynorm='fp32' (got {decode_polynorm!r})"
        elif dict(pn_kw) != RELEASE_PN_KW:
            why = f"the release polynorm options {RELEASE_PN_KW} (got {dict(pn_kw)})"
        elif n_local % TILE or not 1 <= n_local // TILE <= MAX_TILES:
            why = f"a local width of 1..{MAX_TILES} tiles (got {n_local})"
        if why is not None:
            if explicit:
                raise ValueError(f"shared_polynorm='fused' needs {why}")
            return "composite"
    return mode


def _free(*ts) -> None:
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


# ------------------------------------------------------------------------------------------------------------
# Decode program configs (M = 1 tile). Local helper; candidate for model_config (requested shared change).
# ------------------------------------------------------------------------------------------------------------
def _largest_divisor(n: int, cap: int) -> int:
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def decode_matmul_pc(k: int, n: int, grid, *, in0_block_w: Optional[int] = None, fp32_acc: bool = True):
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` for ``[1, 1, 32, k] @ [k, n]`` (one tile row of activations,
    in0 multicast) on ``grid = (x, y)`` cores with N split as ``per_core_N = ceil(n_tiles / cores)``; ``in0_block_w``
    defaults to the largest divisor of ``k / 32`` that is <= 8. The out subblock is the widest of 1..4 (fp32 dest
    acc) dividing ``per_core_N``. Same structure as ``model_config.mcast1d_matmul_pc`` (G6) but allows an uneven N
    split (trailing cores idle) -- what the dense / shared shapes need."""
    gx, gy = int(grid[0]), int(grid[1])
    cores = gx * gy
    k_tiles, n_tiles = k // TILE, n // TILE
    if k % TILE or n % TILE:
        raise ValueError(f"decode matmul dims must be tile multiples, got k={k} n={n}")
    pcn = int(math.ceil(n_tiles / cores))
    if (cores - 1) * pcn >= n_tiles:
        raise ValueError(f"grid {grid} is too large for N = {n_tiles} tiles (per_core_N {pcn})")
    bw = int(in0_block_w) if in0_block_w is not None else _largest_divisor(k_tiles, 8)
    if k_tiles % bw:
        raise ValueError(f"in0_block_w {bw} does not divide K = {k_tiles} tiles")
    cap = 4 if fp32_acc else 8
    sub_w = max(d for d in range(1, cap + 1) if pcn % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=bw,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        out_block_h=1,
        out_block_w=pcn,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


# (kind, matmul) -> (grid, in0_block_w) of the decode matmuls (M = one tile), from the traced sweep on this Galaxy
# (tests/unit/test_mlp.py::test_mlp_device_variants, bfp8 weights, 2026-10-01; auto = ttnn's default config):
#   dense  gate_up [4096, 3072] fp32 out: (12, 8) bw 8  40.1 us (333 GB/s)  | auto 93.5 us
#   dense  down    [1536, 4096] bf16 out: (8, 4)  bw 8  24.7 us (270 GB/s)  | auto 30.0 us
#   shared gate_up [4096, 320]  fp32 out: (10, 1) bw 32 13.2 us             | auto 51.3 us (bw 8: 20.0, bw 16: 15.0)
#   shared down    [160, 4096]  bf16 out: (8, 4)  bw 5   5.2 us             | auto  7.7 us
#   stats="replicated_gate": gate_full [4096, 1280] (10, 4) bw 16 20.0 us (auto 64.5); up [4096, 160] (5, 1) bw 32
#   13.2 us (auto 106.5).
DECODE_MATMUL_GRIDS = {
    ("dense", "gate_up"): ((12, 8), 8),  # N = 96 tiles -> 1 per core
    ("dense", "down"): ((8, 4), 8),  # N = 128 tiles -> 4 per core
    ("shared", "gate_up"): ((10, 1), 32),  # N = 10 tiles -> 1 per core
    ("shared", "down"): ((8, 4), 5),  # N = 128 tiles -> 4 per core
    ("shared", "gate_full"): ((10, 4), 16),  # N = 40 tiles -> 1 per core
    ("shared", "up"): ((5, 1), 32),  # N = 5 tiles -> 1 per core
    ("dense", "gate_full"): ((12, 8), 8),  # [4096, 12288]: N = 384 tiles -> 4 per core (not a sensible choice)
    ("dense", "up"): ((12, 4), 8),  # [4096, 1536]: N = 48 tiles -> 1 per core
}


def build_decode_program_configs(kind: str, dims: dict, compute_grid, grids: Optional[dict] = None):
    """``({matmul: program config | None}, [fallbacks])`` for the decode matmuls ``dims = {name: (k, n)}`` of ``kind``.

    Each tuned ``(grid, in0_block_w)`` of ``grids`` (default :data:`DECODE_MATMUL_GRIDS`) is checked against the chip's
    ``compute_grid`` (``cfg.compute_grid``, read from the device) and the shape: a grid that does not fit (a
    differently harvested chip, or a non-default intermediate size) gives ``None`` = ttnn's auto config, which is
    correct but slower; such matmuls are listed in ``fallbacks``. Matmuls without a tuned entry use the auto config
    (not a fallback). Pure host code (config objects only)."""
    grids = DECODE_MATMUL_GRIDS if grids is None else grids
    cx, cy = (int(v) for v in compute_grid)
    pcs, fallbacks = {}, []
    for mm, (k, n) in dims.items():
        entry = grids.get((kind, mm))
        if entry is None:
            pcs[mm] = None
            continue
        (gx, gy), bw = entry
        if gx > cx or gy > cy:
            pcs[mm] = None
            fallbacks.append(f"{mm}: grid {(gx, gy)} exceeds the compute grid {(cx, cy)}")
            continue
        try:
            pcs[mm] = decode_matmul_pc(k, n, (gx, gy), in0_block_w=bw)
        except ValueError as e:
            pcs[mm] = None
            fallbacks.append(f"{mm}: {e}")
    return pcs, fallbacks


class PolyNormMLP:
    """Dense PolyNorm MLP (layers 0-1) or shared expert (MoE layers) of one layer, TP8 within each DP row.

    Args:
        mesh_device: the opened (4, 8) / (8, 4) mesh.
        cfg: :class:`MotifTTConfig`.
        layer_idx: decoder layer index.
        source: ``weights.HFWeightLoader`` / ``DictWeightSource`` (HF names).
        ccl: :class:`MotifCCL` (``ar_tp`` for the PolyNorm moments and the output).
        kind: "dense" | "shared" (default: from ``cfg.layer(l).is_moe``).
        cache: use the TT weight cache (False for random weights).
        prefix: HF weight prefix (default ``model.layers.{l}.mlp`` / ``model.layers.{l}.moe.shared_experts``).
        weight_dtype: override of the bfp8 policy dtype (``ttnn.bfloat16`` = the fp32-faithful accuracy check).
        decode_polynorm / prefill_polynorm: "fp32" | "bf16" PolyNorm intermediates (design §1.5: fp32 in decode;
            bf16 allowed in prefill if PCC holds).
        decode_program_configs: ``{"gate_up": pc|None, "down": pc|None}`` override (None entries = auto).
        intermediate_memory_config: memory config of decode intermediates (default L1 interleaved; DRAM for prefill).
        shared_polynorm: decode PolyNorm of the shared expert (B5; ``None`` = ``cfg.shared_polynorm``,
            ``MOTIF3_SHARED_POLYNORM``): "composite" (the release) | "fused" (``tt/kernels/shared_polynorm.py``,
            bitwise equal); see :func:`resolve_shared_polynorm`.
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        layer_idx: int,
        *,
        source,
        ccl: MotifCCL,
        kind: Optional[str] = None,
        cache: bool = True,
        prefix: Optional[str] = None,
        weight_dtype=None,
        decode_polynorm: str = "fp32",
        prefill_polynorm: str = "fp32",
        decode_program_configs: Optional[dict] = None,
        intermediate_memory_config=None,
        prefill_row_chunk: Optional[int] = None,
        stats: str = "tp",
        polynorm_options: Optional[dict] = None,
        pad_decode_rows: bool = True,
        shared_polynorm: Optional[str] = None,
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.spec = cfg.layer(self.layer_idx) if self.layer_idx < cfg.num_layers else None
        if kind is None:
            if self.spec is None:
                raise ValueError("kind must be given for a layer outside cfg.num_layers")
            kind = "shared" if self.spec.is_moe else "dense"
        if kind not in MLP_KINDS:
            raise ValueError(f"kind must be one of {MLP_KINDS}, got {kind!r}")
        for m in (decode_polynorm, prefill_polynorm):
            if m not in POLYNORM_MODES:
                raise ValueError(f"polynorm mode must be one of {POLYNORM_MODES}, got {m!r}")
        if stats not in MLP_STATS:
            raise ValueError(f"stats must be one of {MLP_STATS}, got {stats!r}")
        self.stats = stats
        self.kind = kind
        self.ccl = ccl
        self.decode_polynorm = decode_polynorm
        self.prefill_polynorm = prefill_polynorm
        self.hidden = cfg.hidden_size
        self.inter = cfg.intermediate_size if kind == "dense" else cfg.shared_intermediate
        tp = ccl.axis_size("tp")
        if self.inter % (tp * TILE):
            raise ValueError(f"intermediate {self.inter} does not split into tile-aligned TP{tp} shards")
        self.n_local = self.inter // tp  # 1536 dense / 160 shared
        self.prefill_row_chunk = int(prefill_row_chunk or cfg.prefill_row_chunk)
        if self.prefill_row_chunk % TILE:
            raise ValueError(f"prefill_row_chunk {self.prefill_row_chunk} must be a multiple of {TILE}")

        l = self.layer_idx
        check_polynorm_semantics(cfg)
        self.output_scale = polynorm_output_scale(cfg, l)  # x0.5, folded into W_down
        self.prefix = prefix or W.hf_name(l, "mlp" if kind == "dense" else "moe.shared_experts")
        role = "dense_mlp" if kind == "dense" else "shared"
        self.weight_dtype = weight_dtype or (cfg.dtypes.dense_mlp if kind == "dense" else cfg.dtypes.shared_expert)
        cname = ("mlp" if kind == "dense" else "moe.shared") if cache else None
        p = self.prefix

        def gate_up_chip(t, g, u):  # [4096, 2n] = [gate_c | up_c] (weights.mlp_gate_up_for_chip)
            return W.mlp_gate_up_for_chip(g, u, cfg, t)

        def gate_up():
            g = source.get(f"{p}.gate_proj.weight")
            u = source.get(f"{p}.up_proj.weight")
            return W.stack_tp(lambda t: gate_up_chip(t, g, u), cfg, dim=1).reshape(1, 1, self.hidden, -1)

        def up_only():
            g = source.get(f"{p}.gate_proj.weight")
            u = source.get(f"{p}.up_proj.weight")
            n = self.n_local
            return W.stack_tp(lambda t: gate_up_chip(t, g, u)[:, n:], cfg, dim=1).reshape(1, 1, self.hidden, -1)

        def gate_full():  # replicated full gate [4096, I] (same values as the TP shards, concatenated)
            g = source.get(f"{p}.gate_proj.weight")
            return W._f32(g).t().contiguous().reshape(1, 1, self.hidden, -1)

        def down():
            d = source.get(f"{p}.down_proj.weight")
            return W.stack_tp(lambda t: W.mlp_down_for_chip(d, cfg, t, output_scale=self.output_scale), cfg,
                              dim=0).reshape(1, 1, -1, self.hidden)

        def upload(src, name, **kw):
            return W.as_tensor(src, mesh_device=mesh_device, cfg=cfg, dtype=self.weight_dtype,
                               cache_name=f"{cname}.{name}" if cname else None, layer=l, **kw)

        self.w_gate_up = self.w_gate_full = self.w_up = None
        if stats == "tp":
            self.w_gate_up = upload(gate_up, "gate_up", tp_dim=3)  # chip: [1, 1, 4096, 2n] = [gate_c | up_c]
        else:  # "replicated_gate": full gate on every chip (local exact moments), up TP-sharded
            self.w_gate_full = upload(gate_full, "gate_full")  # chip: [1, 1, 4096, I]
            self.w_up = upload(up_only, "up", tp_dim=3)  # chip: [1, 1, 4096, n]
        self.w_down = upload(down, "down", tp_dim=2)  # chip: [1, 1, n, 4096] (x0.5 folded)
        # PolyNorm constants: cached like the weights; act_fn.{weight,bias} is read only on a cache miss (no bias clamp
        # for dense / shared)
        self.pn = ScalarPolyNormConsts(mesh_device, cfg, lambda: PolyNormCoefficients.from_source(source, p),
                                       inter=self.inter, cache_name=f"{cname}.polynorm" if cname else None, layer=l)

        self.ckc_mm = cfg.compute_config(role)
        self.ckc_pn = cfg.compute_config("polynorm")
        self.pn_exact_ar = True  # prefill moments AR on the exact (ROW_MAJOR, all-gather + local sum) path
        # polynorm_tp implementation knobs (tt/polynorm.py): moments / horner / ar
        self.pn_kw = dict(moments="sum", horner="mac", ar="ag_sum")
        self.pn_kw.update(polynorm_options or {})
        self.pad_decode_rows = bool(pad_decode_rows)
        self.decode_imc = intermediate_memory_config if intermediate_memory_config is not None else ttnn.L1_MEMORY_CONFIG
        mms = ("gate_up", "down") if stats == "tp" else ("gate_full", "up", "down")
        self.decode_pc_fallbacks = []
        if decode_program_configs is None:
            decode_program_configs, self.decode_pc_fallbacks = build_decode_program_configs(
                kind, {mm: self._mm_dims(mm) for mm in mms}, cfg.compute_grid
            )
            if self.decode_pc_fallbacks:
                print(f"[motif3.mlp] layer {l} {kind}: decode matmuls on ttnn's auto config: "
                      f"{'; '.join(self.decode_pc_fallbacks)}", flush=True)
        self.decode_pc = dict(decode_program_configs)
        # B5: fused decode PolyNorm of the shared expert (None = the composite)
        self.shared_polynorm = resolve_shared_polynorm(shared_polynorm, cfg, kind=kind, stats=stats,
                                                       decode_polynorm=decode_polynorm, pn_kw=self.pn_kw,
                                                       n_local=self.n_local)
        self.pn_fused = None
        if self.shared_polynorm == "fused":
            from .kernels.shared_polynorm import FusedSharedPolyNorm

            self.pn_fused = FusedSharedPolyNorm(mesh_device, self.pn, ccl, n_local=self.n_local)

    @property
    def coeffs(self) -> PolyNormCoefficients:
        """Host PolyNorm coefficients (lazy: reads the source on first access when the constants came from the cache)."""
        return self.pn.coeffs

    def _mm_dims(self, mm: str):
        return {
            "gate_up": (self.hidden, 2 * self.n_local),
            "gate_full": (self.hidden, self.inter),
            "up": (self.hidden, self.n_local),
            "down": (self.n_local, self.hidden),
        }[mm]

    def _fused_now(self, mode: str, decode: bool, M: int) -> bool:
        """The fused shared PolyNorm runs on this call: decode, one tile row, and the settings it reproduces (checked per
        call: tests switch ``decode_polynorm`` / ``pn_kw`` on a built module)."""
        return (self.pn_fused is not None and decode and M == TILE and mode == "fp32" and self.stats == "tp"
                and self.pn_kw == RELEASE_PN_KW)

    # ---------------------------------------------------------------------------------------------------
    def _rows(self, x, *, mode: str, decode: bool, all_reduce: bool, out_dtype, out_mc, imc, taps=None):
        """gate_up -> PolyNorm (TP or local moments) -> down [-> all_reduce(tp)] on ``x [1, 1, M, 4096]``."""
        n = self.n_local
        M = int(x.shape[-2])
        gu_dtype = ttnn.float32 if mode == "fp32" else ttnn.bfloat16
        pc = (lambda mm: self.decode_pc.get(mm)) if decode else (lambda mm: None)

        def mm(a, w, name, dtype, mc):
            return ttnn.linear(a, w, dtype=dtype, memory_config=mc, compute_kernel_config=self.ckc_mm,
                               program_config=pc(name))

        g_full = g = u = None
        if self._fused_now(mode, decode, M):  # B5: moments kernel -> the release's TP all-gather -> apply kernel
            gu = mm(x, self.w_gate_up, "gate_up", gu_dtype, imc)
            h = self.pn_fused(gu, memory_config=imc)
            if taps is not None:
                g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, M, n], memory_config=imc)
                u = ttnn.slice(gu, [0, 0, 0, n], [1, 1, M, 2 * n], memory_config=imc)
            _free(gu)
        elif self.stats == "tp":
            gu = mm(x, self.w_gate_up, "gate_up", gu_dtype, imc)
            g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, M, n], memory_config=imc)
            u = ttnn.slice(gu, [0, 0, 0, n], [1, 1, M, 2 * n], memory_config=imc)
            _free(gu)
            h = polynorm_tp(g, u, self.pn, ccl=self.ccl, mode=mode, memory_config=imc,
                            compute_kernel_config=self.ckc_pn, exact_ar=self.pn_exact_ar, **self.pn_kw)
        else:
            g_full = mm(x, self.w_gate_full, "gate_full", ttnn.float32, imc)
            u = mm(x, self.w_up, "up", gu_dtype, imc)
            g = self.ccl.partition(g_full, 3, "tp", memory_config=imc)  # this chip's n gate columns
            kw = {k: v for k, v in self.pn_kw.items() if k != "ar"}
            h = polynorm_tp(g, u, self.pn, mode=mode, memory_config=imc, compute_kernel_config=self.ckc_pn,
                            g_stats=g_full, **kw)
        if taps is not None:
            taps["gate"], taps["up"] = g, u
        else:
            _free(g, u)
        _free(g_full)
        y = mm(h, self.w_down, "down", out_dtype, out_mc if not all_reduce else imc)
        if taps is not None:
            taps["act"] = h
        else:
            _free(h)
        if not all_reduce:
            return y
        out = self.ccl.ar_tp(y, memory_config=out_mc)
        _free(y)
        return out

    def forward_decode(self, x, *, all_reduce: bool = True, out_dtype=ttnn.bfloat16, memory_config=None, taps=None):
        """Decode: ``x [1, 1, T, 4096]`` bf16 (T <= 32; 8 lanes of this DP row) -> ``[1, 1, T, 4096]`` ``out_dtype`` in
        DRAM (or ``memory_config``). ``all_reduce=False`` returns this chip's TP partial (the shared expert's
        contribution to the MoE's single ``ar_tp``). ``taps`` (eager tests only): receives gate / up / act.

        Side effect (``pad_decode_rows``, T < 32): rows T..31 of ``x``'s tile padding are zero-filled in place (module
        docstring); ``x``'s logical rows are unchanged and ``x`` is not consumed."""
        T = int(x.shape[-2])
        if T > TILE:
            raise ValueError(f"decode expects at most {TILE} rows, got {T}; use forward_prefill")
        out_mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        if not self.pad_decode_rows or T == TILE:
            return self._rows(x, mode=self.decode_polynorm, decode=True, all_reduce=all_reduce, out_dtype=out_dtype,
                              out_mc=out_mc, imc=self.decode_imc, taps=taps)
        # logical rows = one full tile: no implicit-padding fills inside the reductions / CCLs (profile: 5 + 11 us).
        # ttnn.pad within the tile padding = in-place fill_implicit_tile_padding of x + a view (pad.cpp:478-480): xp
        # aliases x's DRAM buffer (the memory_config below is not applied; it only matters if ttnn.pad ever copies) and
        # x's padding rows T..31 become 0. Never deallocate xp explicitly (that would free the caller's input); dropping
        # the reference is enough. Rows T..31 are don't-care (every op here is row-local) and are sliced off at the end.
        xp = ttnn.pad(x, [(0, 0), (0, 0), (0, TILE - T), (0, 0)], 0.0, memory_config=self.decode_imc)
        y = self._rows(xp, mode=self.decode_polynorm, decode=True, all_reduce=all_reduce, out_dtype=out_dtype,
                       out_mc=self.decode_imc, imc=self.decode_imc, taps=taps)
        del xp
        out = ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, self.hidden], memory_config=out_mc)
        _free(y)
        return out

    def forward_prefill(self, x, *, all_reduce: bool = True, out_dtype=ttnn.bfloat16, memory_config=None, taps=None):
        """Prefill: ``x [1, 1, S, 4096]`` bf16 (S a multiple of 32) -> ``[1, 1, S, 4096]``; rows are processed in
        chunks of ``prefill_row_chunk`` (each chunk all-reduced separately, then concatenated; the last chunk may be
        shorter). Every op is row-local, so the chunked result equals the unchunked one (bitwise on this Galaxy,
        ``test_mlp_device_random_weights``). ``taps`` only for unchunked calls (``S <= prefill_row_chunk``)."""
        S = int(x.shape[-2])
        if S % TILE:
            raise ValueError(f"prefill length {S} must be a multiple of {TILE}")
        out_mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        imc = ttnn.DRAM_MEMORY_CONFIG
        mode = self.prefill_polynorm
        c = self.prefill_row_chunk
        if S <= c:
            return self._rows(x, mode=mode, decode=False, all_reduce=all_reduce, out_dtype=out_dtype, out_mc=out_mc,
                              imc=imc, taps=taps)
        if taps is not None:
            raise ValueError(f"taps are only supported for unchunked prefill (S {S} > prefill_row_chunk {c})")
        outs = []
        for s0 in range(0, S, c):
            s1 = min(S, s0 + c)
            xc = ttnn.slice(x, [0, 0, s0, 0], [1, 1, s1, self.hidden], memory_config=imc)
            outs.append(self._rows(xc, mode=mode, decode=False, all_reduce=all_reduce, out_dtype=out_dtype,
                                   out_mc=imc, imc=imc))
            _free(xc)
        y = ttnn.concat(outs, dim=2, memory_config=out_mc)
        _free(*outs)
        return y

    def __call__(self, x, *, mode: str = "decode", **kw):
        return self.forward_decode(x, **kw) if mode == "decode" else self.forward_prefill(x, **kw)

    def deallocate(self) -> None:
        if self.pn_fused is not None:
            self.pn_fused.deallocate()
        _free(self.w_gate_up, self.w_gate_full, self.w_up, self.w_down)
        self.pn.deallocate()


def MotifDenseMLP(mesh_device, cfg: MotifTTConfig, layer_idx: int, **kw) -> PolyNormMLP:
    """Dense MLP of layer 0 / 1 (``PolyNormMLP(kind="dense")``)."""
    return PolyNormMLP(mesh_device, cfg, layer_idx, kind="dense", **kw)


def MotifSharedExpert(mesh_device, cfg: MotifTTConfig, layer_idx: int, **kw) -> PolyNormMLP:
    """Shared expert of MoE layer ``layer_idx`` (``PolyNormMLP(kind="shared")``)."""
    return PolyNormMLP(mesh_device, cfg, layer_idx, kind="shared", **kw)


__all__ = [
    "DECODE_MATMUL_GRIDS",
    "MLP_KINDS",
    "MLP_STATS",
    "MotifDenseMLP",
    "MotifSharedExpert",
    "PolyNormMLP",
    "RELEASE_PN_KW",
    "build_decode_program_configs",
    "decode_matmul_pc",
    "resolve_shared_polynorm",
]
