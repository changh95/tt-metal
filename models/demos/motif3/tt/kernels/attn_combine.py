# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decode attention combine (Phase C D1 / plan item C3, ``MOTIF3_ATTN_EPILOGUE=fused``).

The absorbed-form decode epilogue (``tt/attention.py`` ``MotifAttention._absorbed_epilogue``) runs, between the
``W_UV`` per-head bmm and ``wo``, six programs on one tile row of lanes::

    u_flat = nlp_concat_heads(u)                     [1, 10, T, 128] -> [1, 1, T, 1280]     (1 core)
    u_sig, noise = nlp_create_q_heads_split(u_flat)  -> [1, 1, T, 1024], [1, 1, T, 256]      (1 core)
    u_noise = noise @ X                              exact 0/1 expansion -> [1, 1, T, 1024]
    d  = addcmul(u_sig, v, u_noise, value=-1.0)      v = sigmoid(lam @ E) (kept: its own matmul)
    dg = multiply(d, g)
    out = where(active, dg, 0.0)

(24.6 us of kernels + about 6 us of dispatch gaps per layer at 32 lanes, logs/opt/phaseC/D1/prof). This module
replaces them with ONE program. The concat / split / noise expansion are pure tile selections: column tile ``j`` of
``u_sig`` is tile ``(h, t)`` of ``u`` (``h = j // 4``, ``t = j % 4``), and column tile ``j`` of ``noise @ X`` is the
noise head of ``h``'s group, tile ``(Sg + h // 4, t)`` (``X`` has one 1 per column, so the matmul copies the value
exactly; it may turn a ``-0.0`` into ``+0.0``, which cannot change any product or sum downstream). The arithmetic runs
the LLK sequence of the three ttnn ops it replaces, with their configuration (``attn_combine/compute.cpp``), so the
output equals the op chain bit for bit (up to the sign of exact zeros), and the ``wo`` output is bitwise unchanged.

Shapes (per chip; bf16 TILE, interleaved; nothing is consumed; ``T`` <= 32 lanes on one tile row)::

    u      [1, Sg + G, T, vdim]     the W_UV output (heads in the virtual order: Sg signal heads, then G noise heads)
    v      [1, 1, T, Sg vdim]       sigmoid(lam @ E)
    g      [1, 1, T, Sg vdim]       sigmoid gate
    active [1, 1, T, Sg vdim]       0/1 lane mask (``MotifAttention.active_mask_*`` with the default width 1024)
    out    [1, 1, T, Sg vdim]       the wo input

Work split: output tile ``j`` on worker ``q = GX y + x``, ``PER`` consecutive tiles per worker (default one tile per
core on an 8 x 4 grid). Trace safety: no host round trip; the output is a fresh allocation; each program (per buffer
layout) is built on the first call and its hash memoized, so traced calls only patch the four input and the output
addresses (common runtime args). L1: 8 two-tile bf16 CBs (32 KB) per worker.
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Dict, Tuple

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "attn_combine"
SOURCES = {name: KERNEL_DIR / f"{name}.cpp" for name in ("reader", "compute", "writer")}

TILE = 32
BF16_TILE_BYTES = 2048
CB_SIG, CB_V, CB_NOISE, CB_G, CB_ACT, CB_D, CB_DG, CB_OUT = 0, 1, 2, 3, 4, 5, 6, 16
CB_TILES = 2
VALUE_BITS = struct.unpack("<I", struct.pack("<f", -1.0))[0]  # addcmul value=-1.0 as the ternary op packs it
DEFAULT_CORES = 32
CORE_CHOICES = (1, 2, 4, 8, 16, 32)


def _sources_tag() -> str:
    h = hashlib.sha1()
    for p in SOURCES.values():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def plan(n_tiles: int, cores: int, grid: Tuple[int, int]) -> Dict[str, int]:
    """Worker layout: ``n_tiles`` output tiles over ``cores`` workers (``PER`` each) on a ``GX x GY`` rectangle that
    fits the compute grid ``grid``. Raises ``ValueError`` when it does not fit."""
    n_tiles, cores = int(n_tiles), int(cores)
    if cores not in CORE_CHOICES:
        raise ValueError(f"fused attention combine: cores {cores} not in {CORE_CHOICES}")
    cores = min(cores, n_tiles)
    per = math.ceil(n_tiles / cores)
    used = math.ceil(n_tiles / per)
    gx = min(int(grid[0]), 8, used)
    gy = math.ceil(used / gx)
    if gy > int(grid[1]):
        raise ValueError(f"fused attention combine: {used} workers do not fit the {grid[0]} x {grid[1]} grid")
    return dict(n=n_tiles, per=per, used=used, gx=gx, gy=gy)


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _is_interleaved(mc) -> bool:
    return mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _cb(index: int, cores):
    return ttnn.CBDescriptor(
        total_size=CB_TILES * BF16_TILE_BYTES,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=BF16_TILE_BYTES)
        ],
    )


def _compute_config():
    """The configuration ternary addcmul / binary_ng multiply / where run bf16 operands with: HiFi4 (the descriptor
    default), fp32 dest acc off, approx off, half-sync DEST; default unpack for the addcmul operands and
    ``UnpackToDestFp32`` for the binary_ng SFPU operands (binary_ng_program_factory sets it on every SFPU input)."""
    from ..model_config import NUM_CB_SLOTS  # lazy: keeps this module's import light

    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = False
    cc.math_approx_mode = False
    cc.dst_full_sync_en = False
    modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    for cb in (CB_D, CB_G, CB_ACT, CB_DG):  # binary_ng SFPU ops (multiply, where) unpack their inputs to DEST in fp32
        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = modes
    return cc


class FusedAttnCombine:
    """The fused decode combine of every attention layer of one mesh (module docstring). Stateless apart from the
    cached program descriptors (one per buffer layout, shared by all layers: the layer's tensors are runtime args).

    Args:
        mesh_device: the mesh.
        signal_heads: ``Sg`` (8 per chip), ``noise_heads``: ``G`` (2), ``vdim``: 128 (multiple of 32).
        cores: workers (``CORE_CHOICES``; default one per output tile, 32).
        debug: 0 (production) | 1 / 2: write ``d`` / ``d * g`` instead of the output (diagnostics).
    """

    def __init__(
        self, mesh_device, *, signal_heads: int, noise_heads: int, vdim: int, cores: int = DEFAULT_CORES, debug: int = 0
    ):
        if int(vdim) % TILE:
            raise ValueError(f"fused attention combine: vdim {vdim} is not a multiple of {TILE}")
        if int(signal_heads) % int(noise_heads):
            raise ValueError(f"fused attention combine: {signal_heads} signal heads over {noise_heads} groups")
        if debug not in (0, 1, 2):
            raise ValueError(f"fused attention combine: debug {debug} not in (0, 1, 2)")
        self.debug = int(debug)  # 0 production; 1 / 2 write d / dg instead (diagnostics)
        self.mesh_device = mesh_device
        self.Sg, self.G, self.vdim = int(signal_heads), int(noise_heads), int(vdim)
        self.tph = self.vdim // TILE
        self.hpg = self.Sg // self.G
        self.width = self.Sg * self.vdim
        g = mesh_device.compute_with_storage_grid_size()
        self.plan = plan(self.Sg * self.tph, cores, (int(g.x), int(g.y)))
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _sources_tag()

    # ---- checks ---------------------------------------------------------------------------------------------------
    def check(self, u, v, g, active) -> int:
        """Raises ``ValueError`` unless the operands have the call contract; returns ``T``."""
        us = tuple(int(x) for x in u.shape)
        if len(us) != 4 or us[0] != 1 or us[1] != self.Sg + self.G or us[3] != self.vdim or not 1 <= us[2] <= TILE:
            raise ValueError(f"fused attention combine: u must be [1, {self.Sg + self.G}, T <= {TILE}, {self.vdim}], "
                             f"got {us}")
        T = us[2]
        for name, t in (("u", u), ("v", v), ("g", g), ("active", active)):
            if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
                raise ValueError(f"fused attention combine: {name} must be bf16 TILE, got {t.dtype} {t.layout}")
            if not _is_interleaved(t.memory_config()):
                raise ValueError(f"fused attention combine: {name} must be interleaved")
            if name != "u" and tuple(int(x) for x in t.shape) != (1, 1, T, self.width):
                raise ValueError(f"fused attention combine: {name} must be [1, 1, {T}, {self.width}], got "
                                 f"{tuple(t.shape)}")
        return T

    def supports(self, u, v, g, active) -> bool:
        try:
            self.check(u, v, g, active)
            return True
        except ValueError:
            return False

    # ---- program --------------------------------------------------------------------------------------------------
    def _cores(self):
        p = self.plan
        full, rem = divmod(p["used"], p["gx"])
        ranges = []
        if full:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(p["gx"] - 1, full - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
        return ttnn.CoreRangeSet(ranges)

    def _program(self, u, v, g, active, out):
        ts = (u, v, g, active, out)
        key = tuple(tuple(_accessor(t)) for t in ts)
        desc = self._desc.get(key)
        if desc is None:
            p = self.plan
            cores = self._cores()
            cbs = [_cb(i, cores) for i in (CB_SIG, CB_V, CB_NOISE, CB_G, CB_ACT, CB_D, CB_DG, CB_OUT)]
            defines = [("MOTIF_ACB_SRC", self._tag)]
            r_ct = ([CB_SIG, CB_V, CB_NOISE, CB_G, CB_ACT, p["n"], self.tph, self.hpg, self.Sg, p["per"], p["gx"]]
                    + _accessor(u) + _accessor(v) + _accessor(g) + _accessor(active))
            c_ct = [CB_SIG, CB_V, CB_NOISE, CB_G, CB_ACT, CB_D, CB_DG, CB_OUT, p["n"], p["per"], p["gx"], VALUE_BITS,
                    self.debug]
            w_ct = [CB_OUT, p["n"], p["per"], p["gx"]] + _accessor(out)

            def kern(name, ct, n_common, config):
                return ttnn.KernelDescriptor(
                    kernel_source=str(SOURCES[name]), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores, compile_time_args=ct, defines=defines, runtime_args=[],
                    common_runtime_args=[0] * n_common, config=config)

            kernels = [
                kern("reader", r_ct, 4, ttnn.ReaderConfigDescriptor()),
                kern("writer", w_ct, 1, ttnn.WriterConfigDescriptor()),
                kern("compute", c_ct, 0, _compute_config()),
            ]
            desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hk = (tuple(r_ct), tuple(c_ct), tuple(w_ct))
                hv = self._hash.get(hk)
                if hv is None:
                    hv = self._hash[hk] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        desc.kernels[0].common_runtime_args = [u.buffer_address(), v.buffer_address(), g.buffer_address(),
                                               active.buffer_address()]
        desc.kernels[1].common_runtime_args = [out.buffer_address()]
        return desc

    # ---- call -----------------------------------------------------------------------------------------------------
    def __call__(self, u, v, g, active, *, memory_config=None):
        """``where(active, (u_sig - v * u_noise) * g, 0)`` ``[1, 1, T, Sg vdim]`` bf16 (``memory_config``, default
        DRAM interleaved). Consumes nothing."""
        T = self.check(u, v, g, active)
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        if not _is_interleaved(mc):
            raise ValueError("fused attention combine: the output must be interleaved")
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, self.width]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.mesh_device, mc)
        ttnn.generic_op([u, v, g, active, out], self._program(u, v, g, active, out))
        return out

    def deallocate(self) -> None:
        self._desc.clear()


__all__ = ["FusedAttnCombine", "plan", "CORE_CHOICES", "DEFAULT_CORES"]
