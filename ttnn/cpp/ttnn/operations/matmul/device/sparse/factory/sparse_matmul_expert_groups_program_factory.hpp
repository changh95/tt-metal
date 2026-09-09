// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation_types.hpp"

namespace ttnn::prim {

// Expert-group parallelism (EGP) program factory of ttnn.sparse_matmul, selected by
// SparseMatmulParams::expert_groups (see its comment). Same block/subblock derivation, CB formats and compute
// kernel as SparseMatmulMultiCoreReuseMcast1DProgramFactory; the grid is G x (output blocks) cores, every core
// decides the validity of every sparsity slot locally (reader_bmm_tile_layout_in0_expert_groups.cpp) and a
// broadcast in0 is multicast once and kept resident in L1.
struct SparseMatmulExpertGroupsProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle in0_kernel_id{};
        tt::tt_metal::KernelHandle in1_kernel_id{};
        std::vector<CoreCoord> cores;
        uint32_t num_cores{};
        uint32_t expert_groups{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ttnn::prim::SparseMatmulParams& operation_attributes,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ttnn::prim::SparseMatmulParams& operation_attributes,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim
