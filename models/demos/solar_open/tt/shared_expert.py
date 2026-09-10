# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Solar-Open always-on shared expert.

``SolarOpenMoE.forward`` returns ``experts(x, idx, w) + shared_experts(x)``: the shared expert is a bias-free
``SolarOpenMLP`` with intermediate size ``moe_intermediate_size * n_shared_experts`` (1280), applied unweighted to the
same post-attention-norm input the router sees.  On the TP mesh this module shards that MLP over its intermediate
dimension exactly like the routed experts (160 columns = 5 tiles per device at TP=8), so its down projection yields a
per-device PARTIAL sum and needs no CCL of its own: the routed experts add the partial to their own pre-all_reduce
partial (``Experts.__call__(..., shared_expert=...)``) and the MoE block's single TP all_reduce reduces both at once.

Program configs (phase 2, 2026-09-07, device profile 5.4): ttnn's auto choice for the ``[T <= 32, 4096] x [4096, 160]``
gate / up linears is a 5-core ``MatmulMultiCoreProgramConfig`` (no multicast, 90 us each per layer = 8.7 ms of the
b1 step); the explicit 1D in0-multicast configs of ``shared_expert_program_configs`` run them in 11 us and the down
linear in 3 us (was 6.9), for row counts up to ``SHARED_EXPERT_CONFIG_MAX_ROWS`` (decode and the traced prefill@128
split). Same fidelity (HiFi2 / bf16 x bfp8), same math; only the K-block accumulation order differs.

With ``MoEOptions.fuse_shared_expert`` the MLP does not build this module: it calls ``load_shared_expert_weights``
(the same three cache stems) and folds the shards into the routed expert tensors as the always-on slot
(``tt/experts/weights.py::fuse_always_on_expert``).

Partial dtype (phase 3e / A3, ``MoEOptions.shared_down_bfp8`` = ``SOLAR_OPEN_SHARED_DOWN_BFP8``): the down linear
emits bf16 by default. With ``decode_down_bfp8`` a DECODE call emits bfloat8_b instead, so the routed experts'
in-place ``ttnn.add(next_states_bfp8, shared, output_tensor=next_states_bfp8)`` becomes a same-dtype add (the
mixed-dtype in-place add on the single-user ``[1, 1, 1, H]`` partial is pathological: 12.6 us per layer at b1, 2.4 us
at b32, ~2 us as bfp8 += bfp8; design_decode_levers.md 2.6 (a)). Not bit-identical: the shared partial is rounded to
bfp8 before instead of after the add. Prefill calls always return bf16 (byte-identical to the previous behaviour).
"""

import ttnn
from models.demos.solar_open.utils.general_utils import get_cache_file_name

from .experts.operations import apply_glu
from .linear_configs import grid_fits, mcast_1d_linear_config

# 1D in0-multicast configs (micro-benchmarked on P150 at M = 1 and 32, identical times; per launch):
#   gate / up [M, 4096] x [4096, 160]: (5, 1) cores x 1 output tile, K in 4 blocks of 32 tiles   90.1 -> 11.1 us
#   down      [M, 160] x [160, 4096]:  (8, 8) cores x 2 output tiles, K = 5 tiles as one block,
#                                      2-tile output subblock                                     6.9 -> 3.0 us
# Rows above SHARED_EXPERT_CONFIG_MAX_ROWS use the auto configs: the per-core in0 block of a 1024-row split would be
# 32 x 32 bf16 tiles = 2 MB. 128 covers decode (1..32 rows, padded to 32 by the batched experts) and the traced
# prefill@128 split (per_core_M 4: in0 block 256 KB double-buffered). 0 disables the configs (A/B switch).
SHARED_EXPERT_GATE_UP_CORES = (5, 1)
SHARED_EXPERT_GATE_UP_IN0_BLOCK_W = 32
SHARED_EXPERT_DOWN_CORES = (8, 8)
SHARED_EXPERT_DOWN_IN0_BLOCK_W = 5  # = Kt for I/tp = 160 (snaps to a divisor of Kt for other TP factors)
SHARED_EXPERT_DOWN_OUT_SUBBLOCK_W = 2
SHARED_EXPERT_CONFIG_MAX_ROWS = 128


def shared_expert_intermediate_size(hf_config, tp):
    """Full shared-expert intermediate width (``moe_intermediate_size * n_shared_experts``), validated to split into
    tile-aligned per-device slices over ``tp`` (Solar-Open: 1280 / 8 = 160 = 5 tiles)."""
    intermediate_size = hf_config.moe_intermediate_size * getattr(hf_config, "n_shared_experts", 1)
    assert intermediate_size > 0 and intermediate_size % (tp * ttnn.TILE_SIZE) == 0, (
        f"shared-expert intermediate {intermediate_size} must split into tile-aligned slices over TP={tp} "
        f"(Solar-Open: 1280 / 8 = 160 = 5 tiles)"
    )
    return intermediate_size


def shared_expert_program_configs(rows, hidden_size, intermediate_per_device, grid, max_rows=None):
    """``(gate_up_config, down_config)`` for ``rows`` logical rows, or ``None`` entries where the auto config stays.

    ``grid`` is the device's compute grid (``CoreCoord``): a config whose core grid does not fit it, or whose N tiles
    do not divide over its cores, falls back to None (auto). ``rows`` above ``max_rows`` (default
    ``SHARED_EXPERT_CONFIG_MAX_ROWS``) return ``(None, None)``.
    """
    max_rows = SHARED_EXPERT_CONFIG_MAX_ROWS if max_rows is None else max_rows
    if rows > max_rows:
        return None, None

    def build(cores, n, k, in0_block_w, out_subblock_w):
        if not grid_fits(cores, grid):
            return None
        try:
            return mcast_1d_linear_config(cores, rows, n, k, in0_block_w, out_subblock_w)
        except ValueError:
            return None

    gate_up = build(
        SHARED_EXPERT_GATE_UP_CORES, intermediate_per_device, hidden_size, SHARED_EXPERT_GATE_UP_IN0_BLOCK_W, 1
    )
    down = build(
        SHARED_EXPERT_DOWN_CORES,
        hidden_size,
        intermediate_per_device,
        SHARED_EXPERT_DOWN_IN0_BLOCK_W,
        SHARED_EXPERT_DOWN_OUT_SUBBLOCK_W,
    )
    return gate_up, down


def load_shared_expert_weights(
    mesh_device, hf_config, state_dict, mesh_config, dtype=ttnn.bfloat8_b, tensor_cache_path=None
):
    """Load (or read from the cache) the TP-sharded shared-expert weights: ``(w_gate, w_up, w_down)``.

    Per device ``w_gate`` / ``w_up`` are ``[1, 1, H, I/tp]`` (column-parallel) and ``w_down`` is ``[1, 1, I/tp, H]``
    (row-parallel) -- the same mappers and shapes as one slot of the routed experts' per-device layout. Cache stems
    (under ``tensor_cache_path``): ``gate_proj_tp{tp}``, ``up_proj_tp{tp}``, ``down_proj_tp{tp}``. Used by
    ``SharedExpert`` and by the always-on expert fusion, so both modes share the cache files.

    Args:
        state_dict: ``substate(mlp_state_dict, "shared_experts")`` = ``gate_proj.weight [I, H]``, ``up_proj.weight
            [I, H]``, ``down_proj.weight [H, I]`` (nn.Linear orientation); ``{}`` loads every tensor from the cache
        dtype: weight dtype (``MoEOptions.shared_expert_dtype``)
    """
    hidden_size = hf_config.hidden_size
    tp = mesh_config.tp
    intermediate_size = shared_expert_intermediate_size(hf_config, tp)

    # nn.Linear stores [out, in]; the device linears consume [in, out] as [1, 1, K, N].
    if state_dict:
        gate = state_dict["gate_proj.weight"].transpose(0, 1).reshape(1, 1, hidden_size, intermediate_size)
        up = state_dict["up_proj.weight"].transpose(0, 1).reshape(1, 1, hidden_size, intermediate_size)
        down = state_dict["down_proj.weight"].transpose(0, 1).reshape(1, 1, intermediate_size, hidden_size)
    else:
        gate = up = down = None

    column_parallel = mesh_config.column_parallel(mesh_device)  # shard the last dim (N) over TP
    row_parallel = mesh_config.row_parallel(mesh_device)  # shard the second-to-last dim (K) over TP

    def load(name, torch_weight, mesh_mapper):
        return ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"{name}_tp{tp}"),
        )

    w_gate = load("gate_proj", gate, column_parallel)  # per device [1, 1, H, I/tp]
    w_up = load("up_proj", up, column_parallel)  # per device [1, 1, H, I/tp]
    w_down = load("down_proj", down, row_parallel)  # per device [1, 1, I/tp, H]
    return w_gate, w_up, w_down


class SharedExpert:

    """``down(silu(gate(x)) * up(x))``, bias-free, TP-sharded over the intermediate dimension.

    Per device: ``w_gate``/``w_up`` ``[1, 1, H, I/tp]`` (column-parallel) and ``w_down`` ``[1, 1, I/tp, H]``
    (row-parallel).  ``__call__`` runs 4 launches (2 linears, one fused SiLU-GLU multiply, the down linear) and returns
    this device's partial over its ``I/tp`` intermediate columns as a bf16 tensor -- the dtype the experts' in-place
    ``ttnn.add(next_states_bfp8, shared_bf16, output_tensor=next_states_bfp8)`` was validated with -- or, with
    ``decode_down_bfp8`` (``MoEOptions.shared_down_bfp8``), as a bfloat8_b tensor for DECODE calls (``partial_dtype``),
    which makes that add a same-dtype op (phase 3e / A3). Prefill partials are bf16 in both cases.

    The three linears run with the explicit 1D in0-multicast program configs of ``shared_expert_program_configs``
    for inputs of up to ``SHARED_EXPERT_CONFIG_MAX_ROWS`` rows (decode, traced prefill@128) and with ttnn's auto
    configs above that; ``program_configs=False`` forces the auto configs everywhere (A/B reference).

    Weight cache stems (under ``<layer>/mlp/shared_experts``): ``gate_proj_tp{tp}``, ``up_proj_tp{tp}``,
    ``down_proj_tp{tp}``.
    """

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        program_configs=True,
        decode_down_bfp8=False,
    ):
        """
        Args:
            mesh_device: TTNN mesh device
            hf_config: ``SolarOpenConfig`` (``hidden_size``, ``moe_intermediate_size``, ``n_shared_experts``,
                ``hidden_act``)
            state_dict: ``substate(mlp_state_dict, "shared_experts")`` = ``gate_proj.weight [I, H]``,
                ``up_proj.weight [I, H]``, ``down_proj.weight [H, I]``; ``{}`` loads every tensor from the cache
            mesh_config: ``MeshConfig`` (provides ``tp`` and the column-/row-parallel mesh mappers)
            dtype: weight dtype (``MoEOptions.shared_expert_dtype``; the output dtype is ``partial_dtype``)
            tensor_cache_path: cache directory for this module's weights (None disables caching)
            program_configs: use the explicit 1D program configs (default); False = ttnn auto configs
            decode_down_bfp8: emit the DECODE down projection (the partial) in bfloat8_b instead of bf16
                (``MoEOptions.shared_down_bfp8``, phase 3e / A3); prefill partials stay bf16
        """
        self.decode_down_bfp8 = bool(decode_down_bfp8)
        activation = getattr(hf_config, "hidden_act", "silu")
        assert activation == "silu", f"SharedExpert implements silu only, hf_config.hidden_act={activation!r}"
        tp = mesh_config.tp
        intermediate_size = shared_expert_intermediate_size(hf_config, tp)
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_size_per_device = intermediate_size // tp
        self.tp = tp

        self.w_gate, self.w_up, self.w_down = load_shared_expert_weights(
            mesh_device, hf_config, state_dict, mesh_config, dtype=dtype, tensor_cache_path=tensor_cache_path
        )

        # bfp8 weights (default): the routed experts' dense-prefill class of config (HiFi2 keeps the full bf16 x bfp8
        # product, bf16 accumulation, L1 packer acc). bf16 weights (SOLAR_OPEN_SHARED_EXPERT_DTYPE=bf16): HiFi4 with
        # fp32 accumulation, otherwise HiFi2 truncates the second bf16 operand and the extra weight precision that
        # option is meant to buy is lost in the matmul (the three linears are tiny: N or K = 160 per device).
        weights_are_bf16 = dtype == ttnn.bfloat16
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4 if weights_are_bf16 else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=weights_are_bf16,
            packer_l1_acc=True,
        )

        # Program configs per logical row count (decode 1 / 32, prefill 128: three entries at most; longer rows map
        # to (None, None) = auto). Cached so traced graphs never rebuild them.
        self.use_program_configs = program_configs
        self._grid = mesh_device.compute_with_storage_grid_size()
        self._program_configs = {}

    def _get_program_configs(self, rows):
        """``(gate_up_config, down_config)`` for ``rows`` logical rows (None entries = ttnn auto config)."""
        if not self.use_program_configs:
            return None, None
        configs = self._program_configs.get(rows)
        if configs is None:
            configs = shared_expert_program_configs(
                rows, self.hidden_size, self.intermediate_size_per_device, self._grid
            )
            self._program_configs[rows] = configs
        return configs

    def partial_dtype(self, is_decode):
        """dtype of the partial ``__call__`` returns: bfloat8_b for a decode call with ``decode_down_bfp8``, else bf16."""
        return ttnn.bfloat8_b if (is_decode and self.decode_down_bfp8) else ttnn.bfloat16

    def __call__(self, x, is_decode=None):
        """Compute this device's partial of the shared-expert output.

        Args:
            x: ``[1, 1, T, H]`` bf16/bfp8 TILE, replicated over the TP axis -- the post-attention-norm hidden states
                (decode: T <= 32 rows, already padded to 32 by the batched experts, ``[1, 1, 1, H]`` for a single
                user; prefill: T = one expert chunk <= 4096 tokens).  Not consumed.
            is_decode: selects L1 (decode) or DRAM (prefill) for the activations; inferred from the row count when
                None.

        Returns:
            ``[1, 1, T, H]`` TILE interleaved, ``partial_dtype(is_decode)`` (bf16; bfloat8_b for a decode call with
            ``decode_down_bfp8``): the partial sum over this device's ``I/tp`` intermediate columns (no CCL; the routed
            experts add it in place into their bfloat8_b partial before their single TP all_reduce).
        """
        rows = x.shape[-2]
        if is_decode is None:
            is_decode = rows <= ttnn.TILE_SIZE
        memory_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        gate_up_config, down_config = self._get_program_configs(rows)
        partial_dtype = self.partial_dtype(is_decode)

        gate = ttnn.linear(
            x,
            self.w_gate,
            dtype=ttnn.bfloat16,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
            program_config=gate_up_config,
        )  # [1, 1, T, I/tp]
        up = ttnn.linear(
            x,
            self.w_up,
            dtype=ttnn.bfloat16,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
            program_config=gate_up_config,
        )  # [1, 1, T, I/tp]
        activated = apply_glu(gate, up)  # up * silu(gate), one binary op
        gate.deallocate(True)
        up.deallocate(True)

        partial = ttnn.linear(
            activated,
            self.w_down,
            dtype=partial_dtype,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
            program_config=down_config,
        )  # [1, 1, T, H], partial over this device's I/tp columns (bf16, or bfp8 at decode with decode_down_bfp8)
        activated.deallocate(True)
        return partial
