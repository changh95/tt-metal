// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/global_circular_buffer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct SparseMatmulParams {
    std::optional<uint32_t> nnz;
    bool is_input_a_sparse;
    bool is_input_b_sparse;
    // When true, an `indices` operand (in optional_input_tensors[0]) drives an indexed/gather mode:
    // the kernels iterate only the `num_active` experts named in `indices` (bB = indices[i]) instead
    // of scanning all batchB sparsity slots, and the output batch axis is COMPACT (length num_active).
    // Set from indices.has_value() at build time so the program hash distinguishes the two modes.
    bool use_indices = false;
    std::optional<const operations::matmul::MatmulProgramConfig> program_config = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    std::optional<const tt::tt_metal::CoreCoord> user_core_coord = std::nullopt;
    std::optional<const tt::tt_metal::Tile> output_tile;
    std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    // Expert-group parallelism (EGP). nullopt (default) selects the legacy mcast_in0 program factory and
    // kernels, byte-for-byte. A value G >= 1 selects SparseMatmulExpertGroupsProgramFactory: the grid is
    // enlarged to G x (output blocks) cores, the non-zero sparsity slots are numbered by their running rank r
    // in scan order and group g = r % G owns slot r, every core decides the validity of every slot locally
    // (no per-slot multicast round trip), and a broadcast in0 ([1,1,M,K] with is_input_b_sparse only) is
    // multicast ONCE and kept resident in L1 instead of being re-read and re-multicast per expert. The
    // per-output-tile math and the output layout are unchanged, so the result is bit-identical to the
    // legacy path. Part of the aggregate, hence of the default program hash.
    std::optional<uint32_t> expert_groups = std::nullopt;
};

struct SparseMatmulInputs {
    std::vector<Tensor> input_tensors;
    std::vector<std::optional<const Tensor>> optional_input_tensors;
    std::vector<std::optional<Tensor>> optional_output_tensors;
};

// EGP kernel zero-fill (the "v2" of the expert-group factory). With `expert_groups` set, the EGP kernels write
// EVERY tile of the output in every run: an expanded scan output gets its non-computed slots (invalid mask
// entries, and ranks >= a caller-supplied nnz) zero-filled by the cores of group `slot % G` with the same tile
// walk as the computed slots, and indexed / compact outputs were already fully written. The device op
// therefore skips the `ttnn::zeros_like` FILL pass it runs before every legacy-path launch (45.5 us on the
// Solar b32 down output, 4.5 us on the gate|up output, per layer). Both the FILL skip (create_output_tensors)
// and the kernel define (EGP factory) are derived from this ONE predicate so they cannot diverge; the factory
// additionally TT_FATALs the coverage preconditions (Mt % per_core_M == 0, interleaved output).
//
// Placement: the zero writes are issued by the in0 EGP reader kernel by default (RISCV_0, the NoC the weight
// stream does not use, idle during the scan apart from the owned slots' in0 pushes), or by the in1 writer
// (the same tile walk, on the streaming RISC) with TT_SPARSE_MATMUL_EGP_ZERO_FILL=in1. A/B knob for
// measurements only, read once per process: TT_SPARSE_MATMUL_EGP_ZERO_FILL=0 restores the phase-3b behaviour
// (FILL kept, no kernel zero-fill), `in1` selects the writer placement, anything else (unset) the in0 reader.
// Not part of the program hash -- fresh process per arm. The legacy factory (`expert_groups == nullopt`) is
// untouched by all of this.
bool sparse_matmul_egp_zero_fill_enabled();
bool sparse_matmul_egp_zero_fill_in_in1_writer();
bool sparse_matmul_egp_kernel_writes_whole_output(const SparseMatmulParams& operation_attributes);

}  // namespace ttnn::prim
