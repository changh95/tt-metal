# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Collective-communication state of the Solar-Open port: the CCLManager (semaphore pools of the ping-pong CCLs) and,
since phase 3e / A2, the fused decode all-reduce behind ``SOLAR_OPEN_DECODE_CCL``.

Decode all-reduce (``SOLAR_OPEN_DECODE_CCL``, read when the CCLManager is built = at model build):

* ``composite``: ``ttnn.all_reduce`` = ReduceScatterMinimalDirect + AllGather, two launches, op-allocated intermediates,
  no persistent state (27.0 us per call at ``[1, 1, 32, 4096]`` bfp8, devices in lockstep; A1). The default until A2.
* ``fused`` (default since phase 3e / A2): ``ttnn.experimental.all_reduce_async`` -- ONE kernel per call: every device's writer multicasts its slices
  into a persistent width-sharded buffer on every output core of every device and increments one global semaphore
  there; the receiver waits for ``ring_size`` increments, resets the semaphore to 0 and sums the 8 slots in a fixed
  order (``fp32_dest_acc``), so the result is deterministic, replica-identical and closer to the fp32 sum than the
  composite's ring accumulation (A1: bfp8 max / mean |err| 0.094 / 0.0187 vs 0.141 / 0.0208; bf16 = the exactly
  rounded sum). Measured 17.2 us per call (Ring, 2 links, bfp8, output width-sharded on the 8x4 decode-norm grid);
  per site incl. the reshards 21.1 (attention) / 19.9 us (MoE) against 30.7 / 27.1 (A1). Integrated (A2, README "Phase 3e
  rows"): traced real layer 0 b1 0.331 -> 0.310 ms, b32 0.796 -> 0.752; demos b1 15.87 -> 14.67 ms/step, b32 37.58 -> 36.64;
  every accuracy floor held (teacher-forced, consistency 0 flips, component and real-weight PCCs within the bfp8 band).

Persistent state (``FusedDecodeAllReducePool``): ``FUSED_NUM_PAIRS`` (buffer, global semaphore) pairs with the FIXED
site -> pair map ``FUSED_SITE_ATTENTION -> 0``, ``FUSED_SITE_MOE -> 1``. The two decode sites alternate (attention, MoE,
attention, ...), and a device sends its slice of a call only after its previous program (the other site's call, which
consumed the other pair) completed, so a pair is never written by a fast device while a slow device still reduces it.
A single pair reused by two consecutive in-flight calls WOULD over-count the semaphore and overwrite the buffer (the
stale-replica hazard class this box has shown with all_gather_async), hence the per-site map -- and the constraint that
one site is never called twice without the other in between while devices may be skewed (the component tests read every
output back to the host between calls, which serializes the devices). The pool is allocated once, BEFORE any trace
capture (``CCLManager.ensure_fused_pool`` at ``Model.__init__``; the tests' first eager call allocates it lazily), the
semaphores are reset to 0 when the model enters decode (``Model.switch_mode``). Prefill keeps ``ttnn.all_reduce``:
Blackhole's kernel refuses a DRAM input and a ``[S, 4096]`` tensor does not fit the buffer.

L1 (per device): 2 x 34,816 B (bfp8) per core on the 32 grid cores (+ 2 x 65,536 B when the bf16 attention arm is on),
a transient 69,632 B scratch CB on the two link-worker cores ((8, 0), (9, 0)) during the call, 4,352 B output shard.
Today's composite needs 272 KB (bfp8) / 512 KB (bf16) of free L1 on EVERY core for its RS-direct staging shard at the
decode shape (A1): with the pool resident that still fits, but nothing else of this size may be added.
"""

import os

from loguru import logger

import ttnn

DECODE_CCL_ENV = "SOLAR_OPEN_DECODE_CCL"
DECODE_CCL_MODES = ("composite", "fused")
DECODE_CCL_DEFAULT = "fused"  # phase 3e / A2: every gate held and the demos are faster (README "Phase 3e rows")

# The two decode all-reduce sites and their persistent (buffer, semaphore) pair (fixed map: pair = site).
FUSED_SITE_ATTENTION = 0
FUSED_SITE_MOE = 1
FUSED_NUM_PAIRS = 2
# Output layout of the fused op: width-sharded on the decode-norm grid (rms_norm.DECODE_NORM_GRID, 8x4 = 32 cores,
# [32, hidden / 32] shards; Solar-Open: [32, 128]). Measured the fastest grid at bfp8 (8x8 1.7x slower, 8x1 4x the L1).
FUSED_DECODE_GRID = (8, 4)
FUSED_DECODE_NUM_LINKS = 2  # 1 link is 1.7x slower (30.3 vs 17.2 us); capped to the compute grid's rows by the op
FUSED_DECODE_TOPOLOGY = ttnn.Topology.Ring  # Linear measured 1.5-3x slower on the 1x8 ring
FUSED_DECODE_FP32_ACC = True  # bit-identical to off on this kernel (fixed slot order) and free; keeps the sum exact
FUSED_DECODE_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b)
FUSED_ROWS = ttnn.TILE_SIZE  # one tile row of users per step ([1, 1, B <= 32, hidden] pads to 32 rows)


def decode_ccl_mode(value=None):
    """``composite`` | ``fused``: ``value`` if given, else ``$SOLAR_OPEN_DECODE_CCL`` (unset / empty = ``DECODE_CCL_DEFAULT``)."""
    mode = (value if value is not None else os.getenv(DECODE_CCL_ENV, "")).strip().lower() or DECODE_CCL_DEFAULT
    if mode not in DECODE_CCL_MODES:
        raise ValueError(f"{DECODE_CCL_ENV}={mode!r} is not one of {DECODE_CCL_MODES}")
    return mode


def fused_decode_grid_fits(hidden_size, compute_grid, grid=FUSED_DECODE_GRID):
    """True when ``hidden_size`` splits into tile-aligned width shards on ``grid`` inside the device's compute grid."""
    gx, gy = grid
    if compute_grid is not None and (gx > compute_grid.x or gy > compute_grid.y):
        return False
    return hidden_size % (gx * gy * ttnn.TILE_SIZE) == 0


def _width_sharded_config(width, grid):
    gx, gy = grid
    return ttnn.create_sharded_memory_config(
        shape=(FUSED_ROWS, width // (gx * gy)),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class FusedDecodeAllReducePool:
    """Persistent buffers + global semaphores of ``ttnn.experimental.all_reduce_async`` at the decode shape
    ``[1, 1, 32, hidden_size]`` on one mesh axis of ``ring_size`` devices (see the module docstring)."""

    def __init__(self, mesh_device, hidden_size, ring_size, grid=FUSED_DECODE_GRID, num_pairs=FUSED_NUM_PAIRS):
        compute_grid = mesh_device.compute_with_storage_grid_size()
        if not fused_decode_grid_fits(hidden_size, compute_grid, grid):
            raise ValueError(
                f"fused decode all-reduce: hidden {hidden_size} does not split into tile-aligned width shards on the "
                f"{grid[0]}x{grid[1]} grid (compute grid {compute_grid.x}x{compute_grid.y})"
            )
        if ring_size % 2:
            raise ValueError(f"fused decode all-reduce: the kernel needs an even ring size, got {ring_size}")
        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.ring_size = ring_size
        self.grid = grid
        self.num_pairs = num_pairs
        self.output_memory_config = _width_sharded_config(hidden_size, grid)
        # buffer shard = ring_size x output shard ([32, 1024] at hidden 4096 / 8 devices): one slot per device
        self.buffer_memory_config = _width_sharded_config(hidden_size * ring_size, grid)
        all_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
        )
        # one global semaphore per pair, on the whole compute grid (the output cores AND the link-worker cores, which
        # lie outside the output grid's bounding box)
        self.semaphores = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(num_pairs)]
        self.buffers = {}  # input dtype -> [num_pairs persistent buffer tensors]
        self.calls = 0

    def ensure(self, dtypes):
        """Allocate the buffer pairs of ``dtypes`` (idempotent). Must run outside any trace capture."""
        for dtype in dtypes:
            if dtype in self.buffers:
                continue
            if dtype not in FUSED_DECODE_DTYPES:
                raise ValueError(f"fused decode all-reduce: unsupported input dtype {dtype}")
            import torch

            self.buffers[dtype] = [
                ttnn.from_torch(
                    torch.zeros(1, 1, FUSED_ROWS, self.hidden_size * self.ring_size),
                    device=self.mesh_device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=self.buffer_memory_config,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                for _ in range(self.num_pairs)
            ]
            logger.info(
                f"fused decode all-reduce pool: {self.num_pairs} x [1, 1, {FUSED_ROWS}, {self.hidden_size * self.ring_size}] "
                f"{dtype} width-sharded buffers on the {self.grid[0]}x{self.grid[1]} grid "
                f"({self.buffer_bytes_per_core(dtype)} B per core each), {self.num_pairs} global semaphores"
            )

    def buffer_bytes_per_core(self, dtype):
        tiles = self.hidden_size * self.ring_size // (self.grid[0] * self.grid[1]) // ttnn.TILE_SIZE
        return tiles * (2048 if dtype == ttnn.bfloat16 else 1088)

    def pair(self, site):
        """The fixed site -> (buffer index, semaphore index) map."""
        return site % self.num_pairs

    def all_reduce(self, x, site, cluster_axis, dtype=None):
        """One fused all-reduce of the width-sharded L1 tensor ``x`` (``[.., 32, hidden_size]`` padded) on pair
        ``site``; returns the sum width-sharded on the output grid in ``dtype`` (default: ``x``'s)."""
        self.ensure((x.dtype,))
        k = self.pair(site)
        self.calls += 1
        compute_grid = self.mesh_device.compute_with_storage_grid_size()
        return ttnn.experimental.all_reduce_async(
            x,
            self.buffers[x.dtype][k],
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=self.semaphores[k],
            dtype=dtype if dtype is not None else x.dtype,
            memory_config=self.output_memory_config,
            topology=FUSED_DECODE_TOPOLOGY,
            num_links=min(FUSED_DECODE_NUM_LINKS, compute_grid.y),
            fp32_dest_acc=FUSED_DECODE_FP32_ACC,
        )

    def reset_semaphores(self):
        for sem in self.semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)

    def release(self):
        for tensors in self.buffers.values():
            for t in tensors:
                t.deallocate(True)
        self.buffers = {}


class CCLManager:
    def __init__(self, mesh_device, num_links, topology=ttnn.Topology.Ring, decode_ccl=None):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        # Decode all-reduce implementation (module docstring): "fused" (all_reduce_async on the persistent pool below,
        # allocated by ensure_fused_pool / the first fused call; the default since A2) or "composite" (ttnn.all_reduce).
        self.decode_ccl = decode_ccl_mode(decode_ccl)
        self._fused_pool = None

        # Cache for ping pong buffers: key = (shape_tuple, dim, mesh_axis), value = [buffer1, buffer2]
        self._ping_pong_buffer_cache = {}
        self._ping_pong_buffer_indices = {}

        # Setup semaphores
        self._init_subdevice()

        # Initialize semaphores for reduce scatter and all gather
        self._init_semaphores()
        self.rs_ping_pong_idx = 0
        self.ag_ping_pong_idx = 0
        self.barrier_idx = 0

    def _init_subdevice(self):
        compute_grid_size = ttnn.CoreCoord(8, 8)
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )

        _worker_sub_device = ttnn.SubDevice(
            [
                self.ccl_cores,
            ]
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)

    def _init_semaphores(self):
        # Initialize semaphores for reduce scatter ping pong
        rs_n_sems = 3 * 2  # 3 semaphores * 2 for ping pong
        self.rs_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)
        ]

        # Initialize semaphores for all gather ping pong
        ag_n_sems = 2 * 2  # 2 semaphores * 2 for ping pong (2 buffers)
        self.ag_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(ag_n_sems)
        ]

        # Initialize barrier semaphores
        barrier_ns_sems = 2 * 1
        self.barrier_semaphore = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(barrier_ns_sems)
        ]

    def get_rs_ping_pong_semaphore(self):
        """
        Get semaphores for reduce scatter ping pong operations.

        Returns:
            List of 3 semaphores for the current ping pong cycle
        """
        cur_idx = self.rs_ping_pong_idx
        n_sems = 3
        self.rs_ping_pong_idx = (cur_idx + 1) % 2
        return self.rs_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_ag_ping_pong_semaphore(self):
        """
        Get semaphores for all gather ping pong operations.

        Returns:
            List of 3 semaphores for the current ping pong cycle
        """
        cur_idx = self.ag_ping_pong_idx
        n_sems = 2
        self.ag_ping_pong_idx = (cur_idx + 1) % 2
        return self.ag_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_barrier_semaphore(self):
        """
        Get semaphores for barrier operations.
        """
        cur_idx = self.barrier_idx
        self.barrier_idx = (cur_idx + 1) % 2
        return self.barrier_semaphore[cur_idx]

    def reset_global_semaphores(self):
        """Reset all global semaphores to 0"""
        for sem in self.rs_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.ag_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        self.reset_fused_semaphores()

    # ------------------------------------------------------------------ fused decode all-reduce (SOLAR_OPEN_DECODE_CCL)
    @property
    def fused_decode_allreduce(self):
        """True when the decode all-reduce sites run ``ttnn.experimental.all_reduce_async`` (knob ``fused``)."""
        return self.decode_ccl == "fused"

    @property
    def fused_pool(self):
        return self._fused_pool

    def ensure_fused_pool(self, hidden_size, dtypes=(ttnn.bfloat8_b,), cluster_axis=1):
        """Allocate the persistent buffers / semaphores of the fused decode all-reduce for ``hidden_size`` (no-op
        with the composite knob). Call at model build, before any trace capture (an allocation inside a capture is a
        forbidden device write); the first fused call allocates lazily for callers that never build a Model."""
        if not self.fused_decode_allreduce:
            return None
        ring_size = tuple(self.mesh_device.shape)[cluster_axis]
        if self._fused_pool is None:
            self._fused_pool = FusedDecodeAllReducePool(self.mesh_device, hidden_size, ring_size)
        elif self._fused_pool.hidden_size != hidden_size or self._fused_pool.ring_size != ring_size:
            raise ValueError(
                f"fused decode all-reduce pool already sized for hidden {self._fused_pool.hidden_size} x ring "
                f"{self._fused_pool.ring_size}, asked for {hidden_size} x {ring_size}"
            )
        self._fused_pool.ensure(dtypes)
        return self._fused_pool

    def fused_decode_applies(self, x, hidden_size=None):
        """True when ``x`` is a decode partial the fused op takes: knob ``fused``, TILE tensor in L1 (the decode
        partials; prefill partials live in DRAM and keep ``ttnn.all_reduce``), bf16 / bfp8, tile-padded shape exactly
        ONE tile row (32 padded rows over the leading dims: ``[1, 1, B <= 32, hidden]``) of ``hidden_size`` columns
        (default: the pool's, else ``x``'s width) that split into tile-aligned shards on the output grid."""
        if not self.fused_decode_allreduce:
            return False
        if x.dtype not in FUSED_DECODE_DTYPES or x.layout != ttnn.TILE_LAYOUT:
            return False
        if x.memory_config().buffer_type != ttnn.BufferType.L1:
            return False
        padded = tuple(x.padded_shape)
        if len(padded) < 2:
            return False
        rows = 1
        for d in padded[:-1]:
            rows *= d
        if rows != FUSED_ROWS:
            return False
        width = padded[-1]
        if hidden_size is None:
            hidden_size = self._fused_pool.hidden_size if self._fused_pool is not None else width
        if width != hidden_size:
            return False
        if self._fused_pool is not None and self._fused_pool.hidden_size != width:
            return False
        return fused_decode_grid_fits(width, self.mesh_device.compute_with_storage_grid_size())

    def fused_decode_all_reduce(self, x, site, cluster_axis=1, dtype=None):
        """Sum the width-sharded L1 decode partial ``x`` over the ``cluster_axis`` devices with the fused kernel on
        pair ``site`` (``FUSED_SITE_ATTENTION`` / ``FUSED_SITE_MOE``); returns the sum width-sharded on the output
        grid (``fused_pool.output_memory_config``), dtype ``dtype`` or ``x``'s. ``x`` is NOT deallocated."""
        pool = self.ensure_fused_pool(tuple(x.padded_shape)[-1], dtypes=(x.dtype,), cluster_axis=cluster_axis)
        return pool.all_reduce(x, site, cluster_axis, dtype=dtype)

    def fused_decode_all_reduce_interleaved(self, x, site, cluster_axis=1, dtype=None, memory_config=None):
        """``fused_decode_all_reduce`` for an interleaved L1 partial: reshards ``x`` onto the output grid, reduces,
        reshards back to ``memory_config`` (default ``ttnn.L1_MEMORY_CONFIG``, today's ``ttnn.all_reduce`` output).
        Deallocates ``x`` (the ``apply_tensor_parallel_allreduce`` contract) and the intermediates."""
        pool = self.ensure_fused_pool(tuple(x.padded_shape)[-1], dtypes=(x.dtype,), cluster_axis=cluster_axis)
        if x.memory_config().is_sharded():
            x_ws = x
        else:
            x_ws = ttnn.to_memory_config(x, pool.output_memory_config)
            x.deallocate(True)
        reduced = pool.all_reduce(x_ws, site, cluster_axis, dtype=dtype)
        x_ws.deallocate(True)
        out = ttnn.to_memory_config(reduced, memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG)
        reduced.deallocate(True)
        return out

    def reset_fused_semaphores(self):
        """Reset the fused pool's global semaphores to 0 (the prefill / decode boundary; never inside a trace)."""
        if self._fused_pool is not None:
            self._fused_pool.reset_semaphores()
