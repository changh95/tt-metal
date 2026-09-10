// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"

#include <cstdlib>
#include <string>
#include <variant>

#include <tt-metalium/work_split.hpp>

namespace {

/**
 * @brief Computes the output shape of a sparse matmul operation given two input tensors.
 *
 * The output shape for a sparse matmul is the same as for a dense matmul, but allows for
 * batching on both input tensors.
 * The final output shape as batched dimensions from input B first (inner), then input A (outer).
 * @param input_tensor_a First input tensor
 * @param input_tensor_b Second input tensor
 * @return Shape of the resulting tensor after sparse matmul
 */
ttnn::Shape compute_sparse_matmul_output_shape(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    bool is_input_a_sparse,
    bool is_input_b_sparse,
    std::optional<uint32_t> num_active = std::nullopt) {
    const auto& input_shape_a = input_tensor_a.logical_shape();
    const auto& input_shape_b = input_tensor_b.logical_shape();

    const auto a_rank = input_shape_a.rank();
    const auto b_rank = input_shape_b.rank();

    // Decide the rank of the output shape based on batch dimensions in input tensors
    // Find batched dimensions in both. Add batched dimensions from both to output rank and then add 2
    // Batched dimensions are all dimensions except the last two
    uint32_t a_batched_dims = ((is_input_a_sparse && is_input_b_sparse) || (a_rank <= 2)) ? 0 : (a_rank - 2);
    uint32_t b_batched_dims = ((is_input_a_sparse && !is_input_b_sparse) || (b_rank <= 2)) ? 0 : (b_rank - 2);
    uint32_t output_rank = a_batched_dims + b_batched_dims + 2;

    // Initialize output shape with zeros based on the output rank
    ttnn::Shape output_shape(std::vector<uint32_t>(output_rank, 0));

    // First pick the M and N dimensions from the input tensors
    output_shape[-2] = input_shape_a[-2];
    output_shape[-1] = input_shape_b[-1];

    // Add batched dims from input B to output shape
    for (uint32_t i = 0; i < b_batched_dims; ++i) {
        output_shape[-3 - i] = input_shape_b[-3 - i];
    }

    // Add batched dims from input A to output shape
    for (uint32_t i = 0; i < a_batched_dims; ++i) {
        output_shape[-3 - b_batched_dims - i] = input_shape_a[-3 - i];
    }

    // Indexed/gather mode: the expert/batch axis is COMPACT (only the num_active selected experts).
    // For every supported mode here input B is sparse with layout [..., E, K, N], so the E batch
    // length lives at output_shape[-3]; overwrite it with num_active. (M/N are unchanged.)
    if (num_active.has_value()) {
        output_shape[-3] = num_active.value();
    }

    return output_shape;
}

ttnn::Shape compute_sparse_matmul_compact_output_shape(
    const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b, uint32_t nnz) {
    return ttnn::Shape{1U, nnz, input_tensor_a.logical_shape()[-2], input_tensor_b.logical_shape()[-1]};
}
}  // namespace

namespace ttnn::prim {

namespace {
// TT_SPARSE_MATMUL_EGP_ZERO_FILL, read once: "0" = off (phase-3b FILL), "in1" = writer placement, else in0 reader.
// The knob is an A/B arm, not a per-call setting (it is not part of the program hash).
const std::string& egp_zero_fill_knob() {
    static const std::string knob = [] {
        const char* env = std::getenv("TT_SPARSE_MATMUL_EGP_ZERO_FILL");
        return std::string(env != nullptr ? env : "");
    }();
    return knob;
}
}  // namespace

bool sparse_matmul_egp_zero_fill_enabled() { return egp_zero_fill_knob() != "0"; }

bool sparse_matmul_egp_zero_fill_in_in1_writer() { return egp_zero_fill_knob() == "in1"; }

bool sparse_matmul_egp_kernel_writes_whole_output(const SparseMatmulParams& operation_attributes) {
    // Expanded scan: the in1 writer zero-fills the non-computed slots (EGP_ZERO_FILL define, set by the factory
    // exactly when this is true and the output is neither indexed nor compact); indexed: every entry i is computed
    // by group i % G; compact static-nnz: every rank < nnz is computed. All three cover every output tile.
    return operation_attributes.expert_groups.has_value() && sparse_matmul_egp_zero_fill_enabled();
}

namespace {
// TT_SPARSE_MATMUL_INDEXED_SKIP_FILL, read once: "0" = keep the FILL on legacy indexed op-owned outputs (phase 3c),
// anything else (unset) = skip it. An A/B arm, not a per-call setting (not part of the program hash).
const std::string& indexed_skip_fill_knob() {
    static const std::string knob = [] {
        const char* env = std::getenv("TT_SPARSE_MATMUL_INDEXED_SKIP_FILL");
        return std::string(env != nullptr ? env : "");
    }();
    return knob;
}
}  // namespace

bool sparse_matmul_indexed_skip_fill_enabled() { return indexed_skip_fill_knob() != "0"; }

bool sparse_matmul_legacy_indexed_kernel_writes_whole_output(
    const SparseMatmulParams& operation_attributes, const SparseMatmulInputs& tensor_args) {
    using namespace operations::matmul::utilities;
    // Legacy factory (sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp) in indexed / gather mode only; the EGP
    // factory has its own predicate above.
    if (operation_attributes.expert_groups.has_value() || !operation_attributes.use_indices) {
        return false;
    }
    if (!sparse_matmul_indexed_skip_fill_enabled()) {
        return false;
    }
    // The coverage proof (types.hpp) needs the writer geometry: an explicit mcast_in0 1D config (the legacy factory
    // derives a config itself when none is given -- keep the FILL there) with Mt % per_core_M == 0 (the h blocks are
    // written without height padding), and the interleaved writer path (the legacy factory never emits OUT_SHARDED).
    if (!operation_attributes.program_config.has_value()) {
        return false;
    }
    const auto* pc = std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
        &operation_attributes.program_config.value());
    if (pc == nullptr || !pc->mcast_in0 || pc->per_core_M == 0) {
        return false;
    }
    if (operation_attributes.output_mem_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    const auto& input_tensor_a = tensor_args.input_tensors.at(0);
    const auto& a_shape_padded = get_matmul_tensor_padded_shape(input_tensor_a, /*transpose=*/false);
    const auto in0_tile = get_matmul_tile(input_tensor_a, /*transpose=*/false);
    const uint32_t Mt = a_shape_padded[-2] / in0_tile.get_height();  // == the factory's get_M_dim(ashape, in0_tile)
    return Mt % pc->per_core_M == 0;
}

void SparseMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void SparseMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace operations::matmul::utilities;
    const auto& input_tensor_a = tensor_args.input_tensors.at(0);
    const auto& input_tensor_b = tensor_args.input_tensors.at(1);
    const auto& sparsity = tensor_args.input_tensors.at(2);

    TT_FATAL(
        input_tensor_a.storage_type() == ttnn::StorageType::DEVICE &&
            input_tensor_b.storage_type() == ttnn::StorageType::DEVICE &&
            sparsity.storage_type() == ttnn::StorageType::DEVICE,
        "All sparse matmul inputs must be on device");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr && input_tensor_b.buffer() != nullptr && sparsity.buffer() != nullptr,
        "All sparse matmul inputs must be allocated in buffers");
    TT_FATAL(
        input_tensor_a.device() == input_tensor_b.device() && input_tensor_a.device() == sparsity.device(),
        "All sparse matmul inputs must be on the same device");
    TT_FATAL(
        input_tensor_a.layout() == ttnn::Layout::TILE,
        "Input tensor A must be TILE layout, got {}",
        input_tensor_a.layout());
    TT_FATAL(
        input_tensor_b.layout() == ttnn::Layout::TILE,
        "Input tensor B must be TILE layout, got {}",
        input_tensor_b.layout());
    TT_FATAL(
        is_floating_point(input_tensor_a.dtype()),
        "Input tensor A must be a floating point type, got {}",
        input_tensor_a.dtype());
    TT_FATAL(
        is_floating_point(input_tensor_b.dtype()),
        "Input tensor B must be a floating point type, got {}",
        input_tensor_b.dtype());
    TT_FATAL(
        sparsity.layout() == ttnn::Layout::ROW_MAJOR,
        "Sparsity tensor must be ROW_MAJOR layout, got {}",
        sparsity.layout());
    TT_FATAL(
        operation_attributes.is_input_a_sparse || operation_attributes.is_input_b_sparse,
        "sparse_matmul requires at least one of is_input_a_sparse or is_input_b_sparse to be true");

    const auto& a_shape_padded = get_matmul_tensor_padded_shape(input_tensor_a, /*transpose=*/false);
    const auto& b_shape_padded = get_matmul_tensor_padded_shape(input_tensor_b, /*transpose=*/false);
    auto in0_tile = get_matmul_tile(input_tensor_a, /*transpose=*/false);
    auto in1_tile = get_matmul_tile(input_tensor_b, /*transpose=*/false);

    TT_FATAL(
        a_shape_padded[-1] == b_shape_padded[-2],
        "Dimension K (A.shape[-1] {}) and B.shape[-2] ({}) must match for A and B",
        a_shape_padded[-1],
        b_shape_padded[-2]);
    TT_FATAL(
        a_shape_padded[-2] % in0_tile.get_height() == 0,
        "a_shape_padded[-2] (A's rows: {}) must be divisible by in0_tile.get_height() (A's tile height: {}) for "
        "tilization. "
        "a_shape_padded: {}, in0_tile: {}",
        a_shape_padded[-2],
        in0_tile.get_height(),
        a_shape_padded,
        in0_tile);
    TT_FATAL(
        a_shape_padded[-1] % in0_tile.get_width() == 0,
        "a_shape_padded[-1] (A's cols: {}) must be divisible by in0_tile.get_width() (A's tile width: {}) for "
        "tilization. "
        "a_shape_padded: "
        "{}, in0_tile: {}",
        a_shape_padded[-1],
        in0_tile.get_width(),
        a_shape_padded,
        in0_tile);
    TT_FATAL(
        b_shape_padded[-2] % in1_tile.get_height() == 0,
        "b_shape_padded[-2] (B's rows: {}) must be divisible by in1_tile.get_height() (B's tile height: {}) for "
        "tilization. "
        "b_shape_padded: {}, in1_tile_shape: {}",
        b_shape_padded[-2],
        in1_tile.get_height(),
        b_shape_padded,
        in1_tile);
    TT_FATAL(
        b_shape_padded[-1] % in1_tile.get_width() == 0,
        "b_shape_padded[-1] (B's cols: {}) must be divisible by in1_tile_shape[1] (B's tile width: {}) for tilization. "
        "b_shape_padded: "
        "{}, in1_tile: {}",
        b_shape_padded[-1],
        in1_tile.get_width(),
        b_shape_padded,
        in1_tile);
    TT_FATAL(
        operation_attributes.nnz.value_or(1) > 0,
        "nnz ({}) must be greater than 0",
        operation_attributes.nnz.value_or(1));

    // Check that nnz is less than or equal to the length of all batch dimensions
    uint32_t batch_length_A = 1;
    if (a_shape_padded.rank() > 2) {
        for (int i = 0; i < a_shape_padded.rank() - 2; ++i) {
            batch_length_A *= a_shape_padded[i];
        }
    }

    uint32_t batch_length_B = 1;
    if (b_shape_padded.rank() > 2) {
        for (int i = 0; i < b_shape_padded.rank() - 2; ++i) {
            batch_length_B *= b_shape_padded[i];
        }
    }

    uint32_t batch_length = 0;
    if (operation_attributes.is_input_a_sparse && operation_attributes.is_input_b_sparse) {
        batch_length = batch_length_B;
    } else if (operation_attributes.is_input_a_sparse) {
        batch_length = batch_length_A;
    } else {
        batch_length = batch_length_A * batch_length_B;
    }

    // Check that sparsity has enough entries
    TT_FATAL(
        sparsity.logical_volume() == batch_length,
        "sparsity logical_volume ({}) must equal batch_length ({}) "
        "[sparsity_shape={}, is_input_a_sparse={}, is_input_b_sparse={}]",
        sparsity.logical_volume(),
        batch_length,
        sparsity.logical_shape(),
        operation_attributes.is_input_a_sparse,
        operation_attributes.is_input_b_sparse);

    TT_FATAL(
        operation_attributes.nnz.value_or(1) <= batch_length,
        "nnz ({}) must be less than or equal to the length of all batch dimensions ({})",
        operation_attributes.nnz.value_or(1),
        batch_length);

    // When nnz is supplied, the receiver and compute kernels loop exactly nnz times while the in0 sender
    // only multicasts once per non-zero sparsity entry. The op therefore requires
    // count_nonzero(sparsity) == nnz; a mismatch deadlocks the device (see issue #45943).
    // count_nonzero(sparsity) is data-dependent and lives on device, so it cannot be checked here on the
    // host -- it is the caller's responsibility to pass an exact nnz, and the contract is validated
    // on-device in reader_bmm_tile_layout_in0_sender_padding.cpp (asserts loudly under watcher instead of
    // hanging).
    // Indexed/gather mode validation. `indices` (optional_input_tensors[0]) is a compacted list of
    // active sparse-group (expert) ids; the kernels iterate it directly (bB = indices[i]) instead of
    // scanning all batch slots, and the output group axis becomes num_active = indices.logical_volume().
    std::optional<uint32_t> indexed_num_active = std::nullopt;
    if (operation_attributes.use_indices) {
        TT_FATAL(
            !tensor_args.optional_input_tensors.empty() && tensor_args.optional_input_tensors.at(0).has_value(),
            "use_indices is set but no indices tensor was provided");
        const auto& indices = tensor_args.optional_input_tensors.at(0).value();
        TT_FATAL(
            operation_attributes.is_input_b_sparse,
            "Indexed/gather mode requires is_input_b_sparse=true (the indexed operand is the expert "
            "weight tensor B, laid out as [..., E, K, N]).");
        // The indices operand is dispatched on input A's device by the program factory, so it must be
        // a device tensor with a live buffer on that same device -- is_allocated() alone is also true
        // for a host tensor, and says nothing about device affinity.
        TT_FATAL(indices.storage_type() == ttnn::StorageType::DEVICE, "indices tensor must be on device");
        TT_FATAL(indices.buffer() != nullptr, "indices tensor must be allocated in a buffer");
        TT_FATAL(
            indices.device() == input_tensor_a.device(),
            "indices tensor must be on the same device as the other sparse matmul inputs");
        TT_FATAL(
            indices.layout() == ttnn::Layout::ROW_MAJOR, "indices must be ROW_MAJOR layout, got {}", indices.layout());
        TT_FATAL(
            indices.dtype() == tt::tt_metal::DataType::UINT16, "indices must be UINT16 dtype, got {}", indices.dtype());
        // The in1 reader fetches the whole id list with a single page-0 read, so every id must live in
        // one ROW_MAJOR stick. A tensor like [1, 1, 8, 1] has the same volume but eight one-element
        // pages, of which only the first would be read.
        const auto& indices_shape = indices.logical_shape();
        TT_FATAL(
            indices_shape[-1] == indices.logical_volume(),
            "indices must occupy a single ROW_MAJOR stick (all dimensions except the last must be 1), got shape {}",
            indices_shape);
        // In indexed mode the loop count comes from num_active, so an nnz would be silently ignored.
        TT_FATAL(
            !operation_attributes.nnz.has_value(),
            "nnz ({}) must not be supplied together with indices: the indexed/gather loop count is "
            "num_active (the length of indices), so nnz would be ignored",
            operation_attributes.nnz.value_or(0));
        // The ids address B's sparse-group axis, and the indexed output shape is the expanded shape
        // with that axis shortened to num_active (see compute_sparse_matmul_output_shape). Both only
        // hold when the group axis is B's *only* batch dimension.
        TT_FATAL(
            b_shape_padded.rank() >= 3,
            "Indexed/gather mode requires input B to have a sparse-group axis (rank >= 3), got shape {}",
            b_shape_padded);
        TT_FATAL(
            batch_length_B == b_shape_padded[-3],
            "Indexed/gather mode requires input B's sparse-group axis to be its only batch dimension "
            "(B batch length {} != B.shape[-3] {}), otherwise the indexed ids and the compact output "
            "shape are ambiguous. B shape: {}",
            batch_length_B,
            b_shape_padded[-3],
            b_shape_padded);
        TT_FATAL(
            indices.logical_volume() <= batch_length_B,
            "indices length / num_active ({}) must be <= the number of sparse groups in input B ({})",
            indices.logical_volume(),
            batch_length_B);
        indexed_num_active = static_cast<uint32_t>(indices.logical_volume());
    }

    const bool is_output_tensor_given =
        !tensor_args.optional_output_tensors.empty() && tensor_args.optional_output_tensors.at(0).has_value();
    if (is_output_tensor_given) {
        // The program factory derives the writer's page geometry from the input tiles
        // ({in0 tile height, in1 tile width}), not from the output tensor, so an optional
        // output with any other tile would be paged differently than the writer writes it.
        const auto& optional_output_tile = tensor_args.optional_output_tensors.at(0)->tensor_spec().tile();
        TT_FATAL(
            optional_output_tile.get_height() == in0_tile.get_height() &&
                optional_output_tile.get_width() == in1_tile.get_width(),
            "Optional output tensor tile {}x{} must match the output tile {}x{} derived from the input tiles",
            optional_output_tile.get_height(),
            optional_output_tile.get_width(),
            in0_tile.get_height(),
            in1_tile.get_width());

        const auto& optional_output_shape = tensor_args.optional_output_tensors.at(0)->logical_shape();
        const auto expanded_output_shape = compute_sparse_matmul_output_shape(
            input_tensor_a,
            input_tensor_b,
            operation_attributes.is_input_a_sparse,
            operation_attributes.is_input_b_sparse);

        if (indexed_num_active.has_value()) {
            // Indexed/gather mode writes num_active compact slots addressed by the loop counter, so the
            // only valid output is the indexed shape. A full-E output would leave holes (the writer no
            // longer zero-fills skipped slots because it never visits them), and an undersized one
            // would be written out of bounds.
            const auto indexed_output_shape = compute_sparse_matmul_output_shape(
                input_tensor_a,
                input_tensor_b,
                operation_attributes.is_input_a_sparse,
                operation_attributes.is_input_b_sparse,
                indexed_num_active);
            TT_FATAL(
                optional_output_shape == indexed_output_shape,
                "Optional output tensor shape {} must match the indexed output shape {} when indices are "
                "provided (num_active={})",
                optional_output_shape,
                indexed_output_shape,
                indexed_num_active.value());
        } else if (operation_attributes.nnz.has_value()) {
            const auto compact_output_shape = compute_sparse_matmul_compact_output_shape(
                input_tensor_a, input_tensor_b, operation_attributes.nnz.value());
            TT_FATAL(
                optional_output_shape == expanded_output_shape || optional_output_shape == compact_output_shape,
                "Optional output tensor shape {} must match expanded output shape {} or compact output shape {} "
                "when nnz={}",
                optional_output_shape,
                expanded_output_shape,
                compact_output_shape,
                operation_attributes.nnz.value());
        } else {
            TT_FATAL(
                optional_output_shape == expanded_output_shape,
                "Optional output tensor shape {} must match expanded output shape {} when nnz is not provided",
                optional_output_shape,
                expanded_output_shape);
        }
    }

    // The mcast_in0 kernel derives in1_num_subblocks = out_block_w / out_subblock_w and bakes it in
    // as a compute-kernel compile-time arg. When out_subblock_w does not divide out_block_w the
    // integer division yields 0, the compute kernel's in1_subblock loop runs zero times and never
    // pushes cb_out, and the in1 writer parks forever on cb_out.wait_front() -- a silent device hang
    // requiring a board reset. The same expression also underflows uint32 in the last-block padded
    // skip count. Reject on the host instead, matching the dense matmul path's
    // validate_matmul_block_and_subblock_configuration.
    //
    // Check order matters: every value used as a divisor below is rejected as zero first, so a
    // malformed program config fails as a TT_FATAL rather than a host divide-by-zero.
    if (operation_attributes.program_config.has_value()) {
        if (const auto* pc = std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
                &operation_attributes.program_config.value())) {
            TT_FATAL(
                pc->out_subblock_w != 0 && pc->out_subblock_h != 0,
                "sparse_matmul: out_subblock_w and out_subblock_h must be non-zero");
            TT_FATAL(
                pc->out_block_w != 0 && pc->out_block_h != 0,
                "sparse_matmul: out_block_w and out_block_h must be non-zero");
            TT_FATAL(
                pc->out_block_w % pc->out_subblock_w == 0,
                "sparse_matmul: out_block_w ({}) must be divisible by out_subblock_w ({}); otherwise "
                "in1_num_subblocks becomes 0 and the mcast_in0 kernel deadlocks",
                pc->out_block_w,
                pc->out_subblock_w);
            TT_FATAL(
                pc->out_block_h % pc->out_subblock_h == 0,
                "sparse_matmul: out_block_h ({}) must be divisible by out_subblock_h ({})",
                pc->out_block_h,
                pc->out_subblock_h);
            TT_FATAL(
                pc->per_core_M % pc->out_block_h == 0,
                "sparse_matmul: per_core_M ({}) must be divisible by out_block_h ({})",
                pc->per_core_M,
                pc->out_block_h);
            TT_FATAL(
                pc->per_core_N % pc->out_block_w == 0,
                "sparse_matmul: per_core_N ({}) must be divisible by out_block_w ({})",
                pc->per_core_N,
                pc->out_block_w);
        }
    }

    // Expert-group parallelism (EGP) contract. Only checked when `expert_groups` is set; the legacy
    // path is untouched. The factory re-derives every quantity below and TT_FATALs again; these
    // are the friendlier, earlier messages.
    if (operation_attributes.expert_groups.has_value()) {
        const uint32_t expert_groups = operation_attributes.expert_groups.value();
        TT_FATAL(expert_groups >= 1, "sparse_matmul: expert_groups must be >= 1 (got {})", expert_groups);
        TT_FATAL(
            expert_groups <= batch_length,
            "sparse_matmul: expert_groups ({}) must not exceed the number of sparse slots ({})",
            expert_groups,
            batch_length);
        TT_FATAL(
            operation_attributes.is_input_b_sparse,
            "sparse_matmul: expert_groups requires is_input_b_sparse=true (the A-sparse/B-dense mode runs on "
            "the legacy path only)");
        // The broadcast-A mode keeps one resident in0 in L1 for the whole op, which needs a single outer A
        // batch; the A+B-sparse mode has batchA == 1 by construction.
        TT_FATAL(
            operation_attributes.is_input_a_sparse || batch_length_A == 1,
            "sparse_matmul: expert_groups requires input A to have a single outer batch ([1, 1, M, K]) when only "
            "B is sparse; got A batch length {}",
            batch_length_A);
        TT_FATAL(
            operation_attributes.program_config.has_value() &&
                std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
                    operation_attributes.program_config.value()),
            "sparse_matmul: expert_groups requires a MatmulMultiCoreReuseMultiCast1DProgramConfig");
        const auto& pc = std::get<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
            operation_attributes.program_config.value());
        TT_FATAL(pc.mcast_in0, "sparse_matmul: expert_groups requires mcast_in0=true");
        const uint32_t Mt = a_shape_padded[-2] / in0_tile.get_height();
        const uint32_t Kt = a_shape_padded[-1] / in0_tile.get_width();
        const uint32_t Nt = b_shape_padded[-1] / in1_tile.get_width();
        TT_FATAL(
            pc.in0_block_w != 0 && Kt % pc.in0_block_w == 0,
            "sparse_matmul: Kt ({}) must be divisible by in0_block_w ({})",
            Kt,
            pc.in0_block_w);
        TT_FATAL(pc.per_core_M != 0 && pc.per_core_N != 0, "sparse_matmul: per_core_M/N must be non-zero");
        // The EGP writer writes per_core_M full tile rows per output block (no height padding), so the h blocks
        // partition the slot's Mt rows only when per_core_M divides Mt. This is what makes "every output tile is
        // written exactly once per run" (the writer zero-fill contract that lets the op skip its FILL pass) hold.
        TT_FATAL(
            Mt % pc.per_core_M == 0,
            "sparse_matmul: expert_groups requires Mt ({}) to be a multiple of per_core_M ({}); the EGP writer "
            "writes whole per_core_M-row blocks (no height padding)",
            Mt,
            pc.per_core_M);
        // The EGP writer addresses the output page by page through an interleaved accessor (no OUT_SHARDED
        // path), and the zero-fill coverage argument is about interleaved pages.
        TT_FATAL(
            operation_attributes.output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "sparse_matmul: expert_groups requires an INTERLEAVED output memory layout, got {}",
            operation_attributes.output_mem_config.memory_layout());
        // The resident in0 ring cycles in (h-block, K-block) order, which matches the compute kernel's
        // consumption order only when at most one of the two per-core block loops has more than one
        // iteration (per_core_M == out_block_h, or per_core_N == out_block_w).
        TT_FATAL(
            operation_attributes.is_input_a_sparse || pc.per_core_M == pc.out_block_h ||
                pc.per_core_N == pc.out_block_w,
            "sparse_matmul: expert_groups with a broadcast A requires per_core_M == out_block_h ({} vs {}) or "
            "per_core_N == out_block_w ({} vs {})",
            pc.per_core_M,
            pc.out_block_h,
            pc.per_core_N,
            pc.out_block_w);
        // The resident in0 is read by one core and multicast ONCE to the whole grid, so every core must need the
        // same in0 rows: a broadcast A must fit one row block (per_core_M == Mt). With several row blocks the
        // cores of rows >= 1 would compute on row block 0's activations (found by the phase-3c M = 128 diagnostic).
        TT_FATAL(
            operation_attributes.is_input_a_sparse || pc.per_core_M == Mt,
            "sparse_matmul: expert_groups with a broadcast A requires per_core_M == Mt ({} vs {}): the resident in0 "
            "is multicast once to the whole grid, so all cores must share the same in0 rows",
            pc.per_core_M,
            Mt);
        const uint32_t num_blocks_total = (((Mt - 1) / pc.per_core_M) + 1) * (((Nt - 1) / pc.per_core_N) + 1);
        const auto grid = pc.compute_with_storage_grid_size;
        const uint32_t num_cores = expert_groups * num_blocks_total;
        TT_FATAL(
            num_cores >= 2,
            "sparse_matmul: expert_groups ({}) x output blocks ({}) must use at least 2 cores",
            expert_groups,
            num_blocks_total);
        TT_FATAL(
            num_cores <= grid.x * grid.y,
            "sparse_matmul: expert_groups ({}) x output blocks ({}) = {} cores exceeds the {}x{} grid",
            expert_groups,
            num_blocks_total,
            num_cores,
            grid.x,
            grid.y);
        // The first num_cores cores in row-major order must fill an exact rectangle (in0 is multicast to
        // the bounding box): either whole rows, or a single partial row.
        const bool exact_rectangle = grid.x > 0 && (num_cores % grid.x == 0 || num_cores < grid.x);
        TT_FATAL(
            exact_rectangle,
            "sparse_matmul: expert_groups ({}) x output blocks ({}) = {} cores does not fill an exact rectangle of "
            "the {}x{} grid (use a grid of {} x ceil({}/{}) or change the number of groups)",
            expert_groups,
            num_blocks_total,
            num_cores,
            grid.x,
            grid.y,
            grid.x,
            num_cores,
            grid.x);
    }
}

SparseMatmulDeviceOperation::spec_return_value_t SparseMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace operations::matmul::utilities;
    TT_FATAL(
        tensor_args.optional_output_tensors.size() <= 1,
        "None or One Optional output tensor can be passed when accessing it "
        "for computing SparseMatmul's output specs");

    const bool is_output_tensor_given =
        !tensor_args.optional_output_tensors.empty() && tensor_args.optional_output_tensors.at(0).has_value();

    if (is_output_tensor_given) {
        return {tensor_args.optional_output_tensors.at(0)->tensor_spec()};
    }

    const auto& input_tensor_a = tensor_args.input_tensors.at(0);
    const auto& input_tensor_b = tensor_args.input_tensors.at(1);

    // Indexed/gather mode -> compact output: the expert axis shrinks from E to num_active (the
    // length of the indices operand carried in optional_input_tensors[0]).
    std::optional<uint32_t> num_active = std::nullopt;
    if (operation_attributes.use_indices && !tensor_args.optional_input_tensors.empty() &&
        tensor_args.optional_input_tensors.at(0).has_value()) {
        num_active = tensor_args.optional_input_tensors.at(0)->logical_volume();
    }

    const auto output_shape = compute_sparse_matmul_output_shape(
        input_tensor_a,
        input_tensor_b,
        operation_attributes.is_input_a_sparse,
        operation_attributes.is_input_b_sparse,
        num_active);

    const auto output_dtype = operation_attributes.output_dtype.has_value() ? operation_attributes.output_dtype.value()
                                                                            : input_tensor_a.dtype();

    auto in0_tile = get_matmul_tile(input_tensor_a, /*transpose=*/false);
    auto in1_tile = get_matmul_tile(input_tensor_b, /*transpose=*/false);

    tt::tt_metal::Tile output_tile = operations::matmul::utilities::get_output_tile(
        operation_attributes.output_mem_config,
        in0_tile,
        in1_tile,
        operation_attributes.output_tile,
        /*optional_output_tensor_tile=*/std::nullopt);

    return {tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            output_dtype,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, output_tile),
            operation_attributes.output_mem_config))};
}

SparseMatmulDeviceOperation::tensor_return_value_t SparseMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    SparseMatmulDeviceOperation::tensor_return_value_t output_tensors;
    const auto& optional_output_tensors = tensor_args.optional_output_tensors;
    const auto& input_tensors = tensor_args.input_tensors;
    // A compact output is fully written by the in1 writer (one block per non-zero sparsity entry),
    // so it skips the zero-fill below; expanded outputs on the LEGACY path rely on it to zero the skipped
    // blocks. On the EGP path (expert_groups set) the kernels write every output tile themselves -- the in1
    // writer zero-fills the non-computed slots of an expanded scan output, indexed and compact outputs are
    // fully written by construction -- so the FILL pass is skipped there (see
    // sparse_matmul_egp_kernel_writes_whole_output in sparse_matmul_device_operation_types.hpp; the EGP factory
    // derives its EGP_ZERO_FILL define from the same predicate and TT_FATALs the coverage preconditions).
    const bool compact_output = !optional_output_tensors.empty() && optional_output_tensors[0].has_value() &&
                                operation_attributes.nnz.has_value() &&
                                optional_output_tensors[0]->logical_shape() ==
                                    compute_sparse_matmul_compact_output_shape(
                                        input_tensors.at(0), input_tensors.at(1), operation_attributes.nnz.value());
    const bool kernel_writes_whole_output = sparse_matmul_egp_kernel_writes_whole_output(operation_attributes);

    if (!optional_output_tensors.empty() and optional_output_tensors[0].has_value()) {
        output_tensors.reserve(optional_output_tensors.size());
        for (const auto& optional_output_tensor : optional_output_tensors) {
            TT_FATAL(
                optional_output_tensor.has_value(),
                "If using optional output tensors, all output tensors must have a value");
            output_tensors.emplace_back(optional_output_tensor.value());
        }
        if (!compact_output && !kernel_writes_whole_output) {
            for (auto& output_tensor : output_tensors) {
                output_tensor = ttnn::zeros_like(
                    output_tensor,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::optional<Tensor>(output_tensor));
            }
        }
        return output_tensors;
    }
    const auto& device = input_tensors.at(0).device();
    const auto& output_specs = compute_output_specs(operation_attributes, tensor_args);
    output_tensors.reserve(output_specs.size());
    for (const auto& output_spec : output_specs) {
        output_tensors.emplace_back(create_device_tensor(output_spec, device));
    }
    // Compact output requires a caller-supplied tensor, so this path is never compact. The op-allocated INDEXED
    // output of the legacy factory is fully written by its in1 writer (every entry, every block, see
    // sparse_matmul_legacy_indexed_kernel_writes_whole_output in sparse_matmul_device_operation_types.hpp), so it
    // skips the FILL too (TT_SPARSE_MATMUL_INDEXED_SKIP_FILL=0 restores it).
    const bool legacy_indexed_fully_written =
        sparse_matmul_legacy_indexed_kernel_writes_whole_output(operation_attributes, tensor_args);
    if (!kernel_writes_whole_output && !legacy_indexed_fully_written) {
        for (auto& output_tensor : output_tensors) {
            output_tensor = ttnn::zeros_like(
                output_tensor,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::optional<Tensor>(output_tensor));
        }
    }
    return output_tensors;
}

// static ttsl::hash::hash_t SparseMatmulDeviceOperation::compute_program_hash(
//     const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

std::tuple<SparseMatmulParams, SparseMatmulInputs> sparse_matmul_build_operation_args(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& sparsity,
    const std::optional<Tensor>& optional_output_tensor,
    std::optional<uint32_t> nnz,
    bool is_input_a_sparse,
    bool is_input_b_sparse,
    const std::optional<const MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreCoord>& user_core_coord,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const std::optional<Tensor>& indices,
    const std::optional<uint32_t>& expert_groups) {
    auto sparse_matmul_attributes = SparseMatmulParams{
        nnz,
        is_input_a_sparse,
        is_input_b_sparse,
        indices.has_value(),  // use_indices
        program_config,
        memory_config.has_value() ? memory_config.value() : ttnn::DRAM_MEMORY_CONFIG,
        dtype,
        compute_kernel_config,
        user_core_coord,
        output_tile,
        global_cb,
        sub_device_id,
        expert_groups};

    auto parameters = create_sparse_matmul_attributes(
        input_tensor_a, input_tensor_b, sparsity, sparse_matmul_attributes, {optional_output_tensor});

    // The indices operand (if any) rides in optional_input_tensors[0]; presence there is the sole
    // trigger for indexed/gather mode. When absent, optional_input_tensors stays empty and every
    // downstream path is byte-for-byte identical to the dense sparsity-scan behavior.
    std::vector<std::optional<const Tensor>> optional_inputs;
    if (indices.has_value()) {
        optional_inputs.emplace_back(indices);
    }

    return {
        parameters,
        SparseMatmulInputs{{input_tensor_a, input_tensor_b, sparsity}, optional_inputs, {optional_output_tensor}}};
}

SparseMatmulDeviceOperation::tensor_return_value_t sparse_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& sparsity,
    const std::optional<Tensor>& optional_output_tensor,
    std::optional<uint32_t> nnz,
    bool is_input_a_sparse,
    bool is_input_b_sparse,
    const std::optional<const MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreCoord>& user_core_coord,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const std::optional<Tensor>& indices,
    const std::optional<uint32_t>& expert_groups) {
    auto [params, inputs] = sparse_matmul_build_operation_args(
        input_tensor_a,
        input_tensor_b,
        sparsity,
        optional_output_tensor,
        nnz,
        is_input_a_sparse,
        is_input_b_sparse,
        memory_config,
        std::move(dtype),
        program_config,
        std::move(compute_kernel_config),
        user_core_coord,
        output_tile,
        global_cb,
        sub_device_id,
        indices,
        expert_groups);
    return ttnn::device_operation::launch<SparseMatmulDeviceOperation>(params, inputs);
}

SparseMatmulParams create_sparse_matmul_attributes(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& /*sparsity*/,
    const SparseMatmulParams& parameters,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    auto matmul_attributes = MatmulParams{
        parameters.program_config,
        /*bcast_batch=*/std::nullopt,
        parameters.output_mem_config,
        parameters.output_dtype,
        parameters.compute_kernel_config,
        /*untilize_out=*/false,
        parameters.user_core_coord,
        /*user_fused_activation=*/std::nullopt,
        /*user_run_batched=*/false,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        parameters.output_tile,
        parameters.global_cb,
        parameters.sub_device_id};

    auto matmul_struct =
        create_matmul_attributes(input_tensor_a, input_tensor_b, matmul_attributes, {optional_output_tensors.at(0)});
    if (matmul_struct.program_config.has_value()) {
        auto device_grid = input_tensor_a.device()->compute_with_storage_grid_size();
        operations::matmul::normalize_program_config(matmul_struct.program_config.value(), device_grid);
    }
    return SparseMatmulParams{
        parameters.nnz,
        parameters.is_input_a_sparse,
        parameters.is_input_b_sparse,
        parameters.use_indices,
        matmul_struct.program_config,
        matmul_struct.output_mem_config,
        matmul_struct.output_dtype,
        matmul_struct.compute_kernel_config,
        matmul_struct.user_core_coord,
        matmul_struct.output_tile,
        matmul_struct.global_cb,
        matmul_struct.sub_device_id,
        parameters.expert_groups};
}

SparseMatmulDeviceOperation::program_factory_t SparseMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    if (operation_attributes.expert_groups.has_value()) {
        return SparseMatmulExpertGroupsProgramFactory{};
    }
    return SparseMatmulMultiCoreReuseMcast1DProgramFactory{};
}
}  // namespace ttnn::prim
