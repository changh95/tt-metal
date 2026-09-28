# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Plumbing stand-ins for the DFlash2 drafter (tt/dflash2_head.py, built separately) with the interfaces the transport
and the prefill hook rely on (tt/aux_hidden.py ``DFlash2ContextPrefillHook``, tt/pd_transfer.py KV groups):

  * ``StubContextProjector(model)`` -- ``project(aux_rep, positions) -> [(K_j, V_j)] * 5``: random FIXED weights
    (seeded, bit-identical in every process) applied on device to the replicated aux rows: per layer j,
    ``K_j = aux[:, j*dim:(j+1)*dim] @ Wk_j``, ``V_j = ... @ Wv_j`` with ``W`` [dim, kv_heads*head_dim] bf16
    column-sharded over the mesh (device d computes kv heads [d*kv/n_dev, (d+1)*kv/n_dev)), HiFi4 fp32-accumulate
    matmuls; a deterministic per-position scale stands in for RoPE. Host bf16 [S, kv_heads, head_dim] per K / V.
  * ``allocate_stub_drafter_kv(model, num_blocks)`` -- the D-side ``DFlash2Drafter.allocate_kv`` stand-in: 5 (k, v)
    paged cache pairs [num_blocks + 1 pad, kv_heads / n_dev, block_size, head_dim] bf16 per device, registered as
    KV group "dflash2" (``pd_transfer.register_kv_group``).
"""
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import aux_hidden as ah
from models.demos.blackhole.qwen36.tt import pd_transfer
from models.demos.blackhole.qwen36.tt.verify_step import _EXACT_MM


class StubContextProjector:
    def __init__(
        self,
        model,
        n_layers=ah.DFLASH2_N_LAYERS,
        kv_heads=ah.DFLASH2_KV_HEADS,
        head_dim=ah.DFLASH2_HEAD_DIM,
        seed=20260928,
    ):
        self.model = model
        self.mesh = model.mesh_device
        self.dim = int(model.args.dim)
        self.n_dev = int(model.num_devices)
        self.n_layers, self.kv_heads, self.head_dim = int(n_layers), int(kv_heads), int(head_dim)
        assert self.kv_heads % self.n_dev == 0, "kv heads must shard evenly over the mesh"
        g = torch.Generator().manual_seed(int(seed))
        width = self.kv_heads * self.head_dim
        mapper = ttnn.ShardTensorToMesh(self.mesh, dim=-1)

        def _w():
            w = (torch.randn(self.dim, width, generator=g) * 0.02).to(torch.bfloat16)
            return ttnn.from_torch(
                w,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        self.wk = [_w() for _ in range(self.n_layers)]
        self.wv = [_w() for _ in range(self.n_layers)]
        self.comp = ttnn.ConcatMeshToTensor(self.mesh, dim=3)

    def project(self, aux, positions):
        """The host-facing entry (DFlash2ContextProjector.project): aux HOST bf16 [S, n_layers*dim] -> uploaded
        replicated, then project_device."""
        aux_t = torch.as_tensor(aux).to(torch.bfloat16).reshape(1, 1, -1, self.n_layers * self.dim)
        aux_tt = ttnn.from_torch(
            aux_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        out = self.project_device(aux_tt, positions)
        ttnn.deallocate(aux_tt)
        return out

    def project_device(self, aux_rep, positions):
        """aux_rep: REPLICATED device bf16 [1,1,S,n_layers*dim]; positions: host int [S] -> [(K, V)] per layer,
        host bf16 [S, kv_heads, head_dim] in global head order (the hook's fast path)."""
        S = int(aux_rep.shape[-2])
        assert int(aux_rep.shape[-1]) == self.n_layers * self.dim, aux_rep.shape
        pos = torch.as_tensor(positions).reshape(-1)[:S].to(torch.int64)
        scale = (1.0 + (pos % 16).float() / 64.0).reshape(S, 1, 1)
        out = []
        for j in range(self.n_layers):
            col = ttnn.slice(aux_rep, (0, 0, 0, j * self.dim), (1, 1, S, (j + 1) * self.dim))
            pair = []
            for w in (self.wk[j], self.wv[j]):
                y = ttnn.matmul(col, w, compute_kernel_config=_EXACT_MM, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                h = ttnn.to_torch(y, mesh_composer=self.comp).to(torch.bfloat16)  # [1,1,S,kv_heads*head_dim]
                ttnn.deallocate(y)
                h = h.reshape(S, self.kv_heads, self.head_dim).float() * scale
                pair.append(h.to(torch.bfloat16).contiguous())
            ttnn.deallocate(col)
            out.append((pair[0], pair[1]))
        return out


def allocate_stub_drafter_kv(
    model,
    num_blocks,
    n_layers=ah.DFLASH2_N_LAYERS,
    kv_heads=ah.DFLASH2_KV_HEADS,
    head_dim=ah.DFLASH2_HEAD_DIM,
    block_size=64,
    dtype=ttnn.bfloat16,
):
    """5 paged (k, v) pairs [num_blocks + 1, kv_heads / n_dev, block_size, head_dim] per device (block ``num_blocks`` =
    the pad block), registered as KV group "dflash2". Allocate BEFORE any trace capture."""
    n_dev = int(model.num_devices)
    assert kv_heads % n_dev == 0
    rep = ttnn.ReplicateTensorToMesh(model.mesh_device)
    shape = [int(num_blocks) + 1, kv_heads // n_dev, int(block_size), int(head_dim)]

    def _mk():
        return ttnn.as_tensor(
            torch.zeros(shape, dtype=torch.bfloat16),
            device=model.mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    pairs = [(_mk(), _mk()) for _ in range(n_layers)]
    return pd_transfer.register_kv_group(model, "dflash2", pairs, pad_block=int(num_blocks))
