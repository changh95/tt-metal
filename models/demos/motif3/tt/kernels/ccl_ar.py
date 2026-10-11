# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase F F5 (``MOTIF3_CCL_FUSE=ar``): the decode all-reduce in ONE program with in-kernel fabric.

Today (Phase E) every decode all-reduce of ``MotifCCL.all_reduce`` with a tile-aligned payload is three programs:

    ttnn.experimental.reduce_scatter_minimal_direct  (RS: every chip unicasts slice j to chip j, chip j sums the N
                                                      blocks in a FIXED order: own slice, then the other chips in
                                                      ascending order, folded pairwise in DST)
    ttnn.all_broadcast (ring, 2 links)               (the race-free "safe" gather of tt/ccl.py)
    ttnn.concat                                       (the N slices -> the full tensor)

:class:`FusedAllReduce` runs the same reduce-scatter (its reader and compute kernels unchanged: ``ccl_ar/reader.cpp``
and ``ccl_ar/compute.cpp`` are byte copies of ttnn's, md5 in :data:`UPSTREAM_MD5`) and does the gather in its writer
(``ccl_ar/writer.cpp``): the reduced slice is written into the full output in place and unicast to every peer's output,
the last packet to a peer carrying a fused atomic inc of that peer's per-source counter on the SAME route (no
alternate-route completion race). The reduction order is the reduce-scatter's, the gather is a copy, so the output is
bitwise the three-program chain's (padding tiles included). Payloads: bf16 TILE interleaved (L1 or DRAM), a ring
axis, the reduce-scatter's direct path (whole-page slices, <= 512 KiB per chip), the scatter dim ``MotifCCL._ar_plan``
picks (the last dim for every decode payload).

Program per mesh coordinate (``ttnn.MeshProgramDescriptor``: the device index, the ring routes and the fabric
connections differ per chip), cached per (axis, input spec, output memory config) with its own global semaphores
(L1_SMALL): reduce arrivals and gather arrivals per source chip, the reader / writer / compute invocation counters and
the start barrier, all monotonic (never reset, as ttnn's). Buffer addresses (input, staging, output) are runtime
arguments patched per call; the staging buffer (both parity halves, as ttnn's) is allocated per call like the
reduce-scatter's, but only on the worker cores.

:class:`FusedAllGather` (``MOTIF3_CCL_FUSE=all``): the race-prone tile-aligned ring gathers (all_broadcast + concat
today, e.g. the shared expert's PolyNorm moments, the sampler gathers) as one program (``ccl_ar/gather.cpp``): the same
start barrier and multicast routes, straight into the concatenated output. Pure data movement (bitwise).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "ccl_ar"
UPSTREAM = "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/device/kernels"
# byte copies of ttnn's reduce_scatter_minimal_direct reader / compute (tt-metal motif3-opt-f): the reduction order.
UPSTREAM_MD5 = {"reader.cpp": "853e14618b8c0108f355150e61fe3704", "compute.cpp": "68a6137321873ae2ed2776a769d48ed7"}
TILE = 32
CB_SEND, CB_REDUCE, CB_OUT = 0, 1, 16
L1_SHARD_BUDGET = 640 << 10  # reduce_scatter_minimal_direct: aliased-CB staging, per worker core
L1_STAGING_BUDGET = 4 << 20  # interleaved-L1 staging fallback, total
DIRECT_RS_MAX_INPUT_BYTES = 512 * 1024


def _ca_size(mesh_device, ca: int) -> int:
    return int(tuple(mesh_device.shape)[ca])


def _ring_routes(mesh, r: int, c: int, ca: int, n: int):
    """The ring of chip (r, c) on cluster axis ``ca`` (n chips): its fabric node, the forward / backward neighbours'
    nodes and every other ring member as a destination (slice = ring index, conn 0 / 1 = the forward / backward
    connection the fabric routes it through, hops = ring distance the shorter way, ties forward; chip / mesh ids for
    2D routing), farthest first (stable; reduce_scatter_minimal_direct_factory.cpp's order and direction fix-up)."""
    me = r if ca == 0 else c

    def at(k):
        return ttnn.MeshCoordinate(*((k, c) if ca == 0 else (r, k)))

    coord = ttnn.MeshCoordinate(r, c)
    self_node = mesh.get_fabric_node_id(coord)
    fwd_node, bwd_node = mesh.get_fabric_node_id(at((me + 1) % n)), mesh.get_fabric_node_id(at((me - 1) % n))
    fwd_dir = ttnn.get_eth_forwarding_direction(self_node, fwd_node)
    bwd_dir = ttnn.get_eth_forwarding_direction(self_node, bwd_node)
    if fwd_dir is None or bwd_dir is None:
        raise RuntimeError(f"ccl_ar: no fabric route to the ring neighbours of {coord}")
    dests = []
    for j in range(n):
        if j == me:
            continue
        fh, bh = (j - me) % n, (me - j) % n
        hops = fh if fh <= bh else bh
        node = mesh.get_fabric_node_id(at(j))
        d = ttnn.get_eth_forwarding_direction(self_node, node)
        if d == fwd_dir:
            conn = 0
        elif d == bwd_dir:
            conn = 1
        else:
            raise RuntimeError(f"ccl_ar: route {coord} -> ring member {j} leaves the ring ({d})")
        dests.append(dict(slice=j, conn=conn, hops=hops, chip=int(node.chip_id), mesh=int(node.mesh_id)))
    dests.sort(key=lambda d: -d["hops"])
    return self_node, fwd_node, bwd_node, dests


class _Prog:
    """One cached mesh workload (per axis / input spec / output memory config)."""

    def __init__(self):
        self.descs: List = []  # per coordinate
        self.ranges: List = []  # its MeshCoordinateRange
        self.sems: List = []  # keep the global semaphores alive
        self.cores: List = []
        self.staging = None  # (shape, mc) of the per-call staging tensor
        self.aliased = False
        self.cb_reduce_bytes = 0


class FusedAllReduce:
    """All-reduce = direct reduce-scatter + in-kernel all-gather, one program (module docstring)."""

    def __init__(self, mesh_device, *, num_links: Optional[int] = None):
        self.mesh_device = mesh_device
        g = mesh_device.compute_with_storage_grid_size()
        self.grid = (int(g.x), int(g.y))
        self.num_links = int(num_links) if num_links else 2
        self._progs: Dict[tuple, _Prog] = {}
        self.calls = 0

    # ---- support -------------------------------------------------------------------------------------------------
    @staticmethod
    def supported(x, n: int, dim: int, out_mc) -> bool:
        """bf16 TILE interleaved rank-4 input, a 4 / 8-chip axis, ``dim`` the scatter dim (whole-page slices), the
        direct reduce-scatter's size limit, an interleaved output (the caller also checks the ring axis)."""
        try:
            if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT or int(n) not in (2, 4, 8):
                return False
            if x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                return False
            if out_mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                return False
            s = [int(d) for d in x.shape]
            p = [int(d) for d in x.padded_shape]
            if len(p) != 4 or dim not in (0, 1, 2, 3):
                return False
            if dim < 2 and p[dim] != s[dim]:
                return False
            if dim >= 2 and p[dim] != s[dim]:
                return False
            pages = p[dim] // TILE if dim >= 2 else p[dim]
            if pages % n:
                return False
            return math.prod(p) // (TILE * TILE) * 2048 <= DIRECT_RS_MAX_INPUT_BYTES
        except Exception:
            return False

    # ---- geometry (reduce_scatter_minimal_direct_factory.cpp: reduce_scatter_direct_geometry) ----------------------
    @staticmethod
    def geometry(padded, dim: int, n: int, tile_bytes: int = 2048):
        rank = len(padded)

        def dim_pages(d):
            if d == rank - 1:
                return padded[d] // TILE
            if d == rank - 2:
                return padded[d] // TILE
            return padded[d]

        dsp = dim_pages(dim)
        inner = 1
        for d in range(dim + 1, rank):
            inner *= dim_pages(d)
        num_pages = math.prod(padded) // (TILE * TILE)
        max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        ipp = max_payload // tile_bytes
        tg = min(4 * min(4, ipp), 8)
        pps = num_pages // n
        return dict(tile_bytes=tile_bytes, tg=tg, ipp=ipp, page_bytes=tg * tile_bytes, pps=pps,
                    cps=(pps + tg - 1) // tg, run=(dsp // n) * inner, stride=dsp * inner)

    # ---- program build -----------------------------------------------------------------------------------------------
    def _build(self, x, ca: int, n: int, dim: int, out_mc) -> _Prog:
        mesh = self.mesh_device
        R, C = (int(v) for v in tuple(mesh.shape))
        padded = [int(d) for d in x.padded_shape]
        geo = self.geometry(padded, dim, n)
        tb, tg, cps, pps = geo["tile_bytes"], geo["tg"], geo["cps"], geo["pps"]
        num_links = min(max(1, self.num_links), cps)
        cores = [ttnn.CoreCoord(i % self.grid[0], i // self.grid[0]) for i in range(num_links)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        base_chunks, extra_chunks = divmod(cps, num_links)
        cc_max = base_chunks + (1 if extra_chunks else 0)

        prog = _Prog()
        prog.cores = cores
        # staging (reduce_scatter_direct_staging_spec), on the worker cores only
        shard_rows = 2 * cps * n
        half_stride = cps * n * tg
        if shard_rows * geo["page_bytes"] <= L1_SHARD_BUDGET:
            prog.aliased = True
            ss = ttnn.ShardSpec(core_set, [shard_rows, geo["page_bytes"]], ttnn.ShardOrientation.ROW_MAJOR)
            smc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ss)
            prog.staging = ([num_links * shard_rows, geo["page_bytes"]], smc)
            cb_reduce_pages = 2 * half_stride
        else:
            total = 2 * n * cps * geo["page_bytes"]
            bt = ttnn.BufferType.L1 if total <= L1_STAGING_BUDGET else ttnn.BufferType.DRAM
            prog.staging = ([2 * n * cps, geo["page_bytes"]],
                            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, bt))
            cb_reduce_pages = 2 * n * tg
        parity = half_stride if prog.aliased else 0
        prog.cb_reduce_bytes = cb_reduce_pages * tb
        staging_t = self._alloc_staging(prog)
        out_t = self._alloc_out(x, out_mc)

        # global semaphores (L1_SMALL): arrivals[n], gather[n], reader_gen, writer_gen, compute_gen, init
        bt = ttnn.BufferType.L1_SMALL
        sems = [ttnn.create_global_semaphore(mesh, core_set, 0, bt) for _ in range(2 * n + 4)]
        ttnn.synchronize_device(mesh)
        prog.sems = sems
        addr = [int(ttnn.get_global_semaphore_address(s)) for s in sems]
        arr, ag = addr[0:n], addr[n:2 * n]
        reader_gen, writer_gen, compute_gen, init_sem = addr[2 * n:2 * n + 4]

        def cbd(idx, pages):
            return ttnn.CBDescriptor(total_size=pages * tb, core_ranges=core_set, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=tb)])

        r_ct = [tb, tg, cps, pps, geo["run"], geo["stride"], n, CB_SEND, CB_REDUCE, int(prog.aliased), parity]
        r_ct += list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        r_ct += list(ttnn.TensorAccessorArgs(staging_t).get_compile_time_args())
        w_ct = [tb, tg, cps, pps, n, geo["ipp"], CB_SEND, CB_OUT, int(prog.aliased), parity, 1, geo["run"],
                geo["stride"], 1]
        w_ct += list(ttnn.TensorAccessorArgs(staging_t).get_compile_time_args())
        w_ct += list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args())
        c_ct = [tg, n, CB_REDUCE, CB_OUT, parity]

        exp = int(os.environ.get("MOTIF3_CCLAR_EXP", "0") or 0)  # timing experiments only (writer.cpp)
        defines = [("MOTIF_CCLAR_EXP", str(exp))] if exp else []
        for r in range(R):
            for c in range(C):
                coord = ttnn.MeshCoordinate(r, c)
                me = r if ca == 0 else c
                desc = ttnn.ProgramDescriptor(
                    kernels=[
                        ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "reader.cpp"),
                                              source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                              core_ranges=core_set, compile_time_args=r_ct,
                                              config=ttnn.ReaderConfigDescriptor()),
                        ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "writer.cpp"),
                                              source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                              core_ranges=core_set, compile_time_args=w_ct, defines=defines,
                                              config=ttnn.WriterConfigDescriptor()),
                        ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "compute.cpp"),
                                              source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                              core_ranges=core_set, compile_time_args=c_ct,
                                              config=ttnn.ComputeConfigDescriptor()),
                    ],
                    semaphores=[],
                    cbs=[cbd(CB_SEND, 2 * tg), cbd(CB_OUT, max(2, cc_max) * tg), self._cb_reduce(prog, staging_t)],
                )

                self_node, fwd_node, bwd_node, dests = _ring_routes(mesh, r, c, ca, n)
                uses_bwd = any(d["conn"] == 1 for d in dests)
                mrange = [sum(1 for d in dests if d["conn"] == k) for k in (0, 1)]
                for link, core in enumerate(cores):
                    chunk_start = link * base_chunks + min(link, extra_chunks)
                    chunk_count = base_chunks + (1 if link < extra_chunks else 0)
                    tile_start = chunk_start * tg
                    tile_count = min(chunk_count * tg, pps - tile_start)
                    peer = mesh.worker_core_from_logical_core(core)
                    rd = [0, 0, me, chunk_start, chunk_count, tile_start, tile_count, reader_gen]
                    rd += [arr[s] for s in range(n) if s != me]
                    rd += [d["slice"] for d in dests]
                    desc.kernels[0].runtime_args[core.x][core.y] = rd
                    desc.kernels[2].runtime_args[core.x][core.y] = [chunk_count, tile_count, compute_gen]
                    wr = [0, 0, me, chunk_start, chunk_count, tile_start, tile_count, writer_gen, arr[me],
                          int(peer.x), int(peer.y), 2 if uses_bwd else 1, init_sem, mrange[0], mrange[1], ag[me]]
                    wr += [ag[s] for s in range(n) if s != me]
                    wr += [d["conn"] for d in dests]
                    wr += [d["hops"] for d in dests]
                    wr += [(me + 1 if me < d["slice"] else me) for d in dests]
                    wr += [d["chip"] for d in dests]
                    wr += [d["mesh"] for d in dests]
                    dst_nodes = [fwd_node] + ([bwd_node] if uses_bwd else [])
                    fab = ttnn.setup_routing_plane_connection(self_node, dst_nodes, [link] * len(dst_nodes), desc, 1,
                                                              core)
                    desc.kernels[1].runtime_args[core.x][core.y] = wr + list(fab)
                if hasattr(ttnn, "compute_program_descriptor_hash"):
                    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
                prog.descs.append(desc)
                prog.ranges.append(ttnn.MeshCoordinateRange(coord, coord))
        ttnn.deallocate(out_t)
        ttnn.deallocate(staging_t)
        return prog

    def _alloc_out(self, x, mc):
        return ttnn.allocate_tensor_on_device(ttnn.Shape([int(d) for d in x.shape]), x.dtype, x.layout,
                                              self.mesh_device, mc)

    def _alloc_staging(self, prog: _Prog):
        shape, mc = prog.staging
        return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, mc)

    def _cb_reduce(self, prog: _Prog, staging_t):
        tb = 2048
        if prog.aliased:
            cb = ttnn.cb_descriptor_from_sharded_tensor(CB_REDUCE, staging_t)
            cb.total_size = prog.cb_reduce_bytes
            cb.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=CB_REDUCE, data_format=ttnn.bfloat16,
                                                             page_size=tb)]
            return cb
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in prog.cores])
        return ttnn.CBDescriptor(total_size=prog.cb_reduce_bytes, core_ranges=core_set, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_REDUCE, data_format=ttnn.bfloat16, page_size=tb)])

    # ---- call --------------------------------------------------------------------------------------------------------
    def __call__(self, x, cluster_axis: int, dim: int, *, memory_config=None):
        """``x`` summed over ``cluster_axis`` (every chip of the axis gets the identical result), in
        ``memory_config`` (DRAM interleaved by default). Consumes nothing."""
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        ca = int(cluster_axis)
        n = _ca_size(self.mesh_device, ca)
        key = (ca, int(dim), tuple(int(d) for d in x.shape), tuple(int(d) for d in x.padded_shape),
               str(x.memory_config()), str(mc))
        prog = self._progs.get(key)
        if prog is None:
            if not self.supported(x, n, int(dim), mc):
                raise ValueError(f"ccl_ar: unsupported all-reduce {x.dtype} {x.layout} {list(x.shape)} dim {dim} "
                                 f"over {n} chips -> {mc}")
            prog = self._progs[key] = self._build(x, ca, n, int(dim), mc)
        staging = self._alloc_staging(prog)
        out = self._alloc_out(x, mc)
        xa, sa, oa = int(x.buffer_address()), int(staging.buffer_address()), int(out.buffer_address())
        md = ttnn.MeshProgramDescriptor()
        for rng, desc in zip(prog.ranges, prog.descs):
            for core in prog.cores:
                ra = desc.kernels[0].runtime_args[core.x][core.y]
                ra[0], ra[1] = xa, sa
                wa = desc.kernels[1].runtime_args[core.x][core.y]
                wa[0], wa[1] = sa, oa
            if prog.aliased:
                cbs = list(desc.cbs)
                cbs[2] = self._cb_reduce(prog, staging)
                desc.cbs = cbs
            md[rng] = desc  # (a copy)
        ttnn.generic_op([x, staging, out], md)
        ttnn.deallocate(staging)
        self.calls += 1
        return out


_TILE_BYTES = {"BFLOAT16": 2048, "FLOAT32": 4096}
GATHER_MAX_TILES_PER_CORE = 64  # the input tiles a gather worker stages in L1


class FusedAllGather:
    """Phase F F5 (``MOTIF3_CCL_FUSE=all``): the race-free ring all-gather of a tile-aligned TILE tensor (bf16 / fp32,
    interleaved) as ONE program (``ccl_ar/gather.cpp``), replacing ``all_broadcast`` + ``concat`` (tt/ccl.py
    ``_all_gather_safe``); pure data movement, bitwise the concat. One program per mesh coordinate (the routes and
    fabric connections differ), cached per (axis, dim, input spec, output memory config) with its own L1_SMALL global
    semaphores (per-source gather counters, the invocation counter and the start barrier; monotonic)."""

    def __init__(self, mesh_device, *, num_links: Optional[int] = None):
        self.mesh_device = mesh_device
        g = mesh_device.compute_with_storage_grid_size()
        self.grid = (int(g.x), int(g.y))
        self.num_links = int(num_links) if num_links else 2
        self._progs: Dict[tuple, _Prog] = {}
        self.calls = 0

    @staticmethod
    def _dt(x) -> str:
        dt = x.dtype
        return getattr(dt, "name", str(dt).split(".")[-1])

    @classmethod
    def supported(cls, x, n: int, dim: int, out_mc) -> bool:
        """TILE bf16 / fp32 interleaved rank-4 input whose gather dim is not tile-padded, a 4 / 8-chip axis, an
        interleaved output, at most 2 x GATHER_MAX_TILES_PER_CORE input tiles per chip."""
        try:
            if x.layout != ttnn.TILE_LAYOUT or cls._dt(x) not in _TILE_BYTES or int(n) not in (2, 4, 8):
                return False
            if x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                return False
            if out_mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                return False
            s = [int(d) for d in x.shape]
            p = [int(d) for d in x.padded_shape]
            d = int(dim) % 4
            if len(p) != 4 or p[d] != s[d]:
                return False
            return math.prod(p) // (TILE * TILE) <= 2 * GATHER_MAX_TILES_PER_CORE
        except Exception:
            return False

    def _build(self, x, ca: int, n: int, dim: int, out_mc) -> _Prog:
        mesh = self.mesh_device
        R, C = (int(v) for v in tuple(mesh.shape))
        p = [int(d) for d in x.padded_shape]
        d = dim % 4
        tb = _TILE_BYTES[self._dt(x)]
        dim_pages = p[d] // TILE if d >= 2 else p[d]
        inner = 1
        for k in range(d + 1, 4):
            inner *= p[k] // TILE if k >= 2 else p[k]
        run = dim_pages * inner
        stride = n * run
        tiles = math.prod(p) // (TILE * TILE)
        num_links = max(1, min(self.num_links, tiles))
        base, extra = divmod(tiles, num_links)
        per_max = base + (1 if extra else 0)
        if per_max > GATHER_MAX_TILES_PER_CORE:
            raise ValueError(f"ccl_ar gather: {tiles} tiles per chip exceed the per-core staging")
        pairs = int(2 * tb <= int(ttnn.get_tt_fabric_max_payload_size_bytes()))
        cores = [ttnn.CoreCoord(i % self.grid[0], i // self.grid[0]) for i in range(num_links)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        prog = _Prog()
        prog.cores = cores
        oshape = [int(v) for v in x.shape]
        oshape[d] *= n
        prog.staging = oshape  # (the output shape)
        out_t = ttnn.allocate_tensor_on_device(ttnn.Shape(oshape), x.dtype, x.layout, mesh, out_mc)
        sems = [ttnn.create_global_semaphore(mesh, core_set, 0, ttnn.BufferType.L1_SMALL) for _ in range(n + 2)]
        ttnn.synchronize_device(mesh)
        prog.sems = sems
        addr = [int(ttnn.get_global_semaphore_address(s)) for s in sems]
        ag, gen, init_sem = addr[0:n], addr[n], addr[n + 1]
        ct = [tb, n, run, stride, pairs, 0]
        ct += list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        ct += list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args())
        exp = int(os.environ.get("MOTIF3_CCLAR_EXP", "0") or 0)  # timing experiments only (gather.cpp)
        defines = [("MOTIF_CCLAR_EXP", str(exp))] if exp else []
        cb = ttnn.CBDescriptor(total_size=per_max * tb, core_ranges=core_set, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=x.dtype, page_size=tb)])
        for r in range(R):
            for c in range(C):
                coord = ttnn.MeshCoordinate(r, c)
                me = r if ca == 0 else c
                desc = ttnn.ProgramDescriptor(
                    kernels=[ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "gather.cpp"),
                                                   source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                                   core_ranges=core_set, compile_time_args=ct, defines=defines,
                                                   config=ttnn.WriterConfigDescriptor())],
                    semaphores=[], cbs=[cb])
                self_node, fwd_node, bwd_node, dests = _ring_routes(mesh, r, c, ca, n)
                uses_bwd = any(dd["conn"] == 1 for dd in dests)
                mrange = [sum(1 for dd in dests if dd["conn"] == k) for k in (0, 1)]
                for link, core in enumerate(cores):
                    t0 = link * base + min(link, extra)
                    cnt = base + (1 if link < extra else 0)
                    peer = mesh.worker_core_from_logical_core(core)
                    rt = [0, 0, me, t0, cnt, gen, init_sem, ag[me], int(peer.x), int(peer.y), 2 if uses_bwd else 1,
                          mrange[0], mrange[1]]
                    rt += [ag[s] for s in range(n) if s != me]
                    dst_nodes = [fwd_node] + ([bwd_node] if uses_bwd else [])
                    fab = ttnn.setup_routing_plane_connection(self_node, dst_nodes, [link] * len(dst_nodes), desc, 0,
                                                              core)
                    desc.kernels[0].runtime_args[core.x][core.y] = rt + list(fab)
                if hasattr(ttnn, "compute_program_descriptor_hash"):
                    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
                prog.descs.append(desc)
                prog.ranges.append(ttnn.MeshCoordinateRange(coord, coord))
        ttnn.deallocate(out_t)
        return prog

    def __call__(self, x, cluster_axis: int, dim: int, *, memory_config=None):
        """Concatenate the shards of ``cluster_axis`` along ``dim`` (ring index order), into ``memory_config`` (DRAM
        interleaved by default). Consumes nothing."""
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        ca = int(cluster_axis)
        n = _ca_size(self.mesh_device, ca)
        d = int(dim) % len(x.shape)
        key = (ca, d, tuple(int(v) for v in x.shape), tuple(int(v) for v in x.padded_shape), self._dt(x),
               str(x.memory_config()), str(mc))
        prog = self._progs.get(key)
        if prog is None:
            if not self.supported(x, n, d, mc):
                raise ValueError(f"ccl_ar gather: unsupported {x.dtype} {x.layout} {list(x.shape)} dim {dim} over {n}")
            prog = self._progs[key] = self._build(x, ca, n, d, mc)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(prog.staging), x.dtype, x.layout, self.mesh_device, mc)
        xa, oa = int(x.buffer_address()), int(out.buffer_address())
        md = ttnn.MeshProgramDescriptor()
        for rng, desc in zip(prog.ranges, prog.descs):
            for core in prog.cores:
                ra = desc.kernels[0].runtime_args[core.x][core.y]
                ra[0], ra[1] = xa, oa
            md[rng] = desc
        ttnn.generic_op([x, out], md)
        self.calls += 1
        return out

