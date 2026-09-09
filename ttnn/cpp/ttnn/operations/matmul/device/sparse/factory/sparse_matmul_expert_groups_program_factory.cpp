// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/sparse/factory/sparse_matmul_expert_groups_program_factory.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"

#include "tt-metalium/work_split.hpp"
#include <tt-metalium/kernel_types.hpp>
#include "tt-metalium/tensor_accessor_args.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

#include <cstdlib>

namespace ttnn::prim {

namespace {

// Debug knob for A/B measurements only: when set, every core reads the resident in0 itself instead of the
// sender multicasting it once (the EGP "v0" data path). Not part of the program hash -- use in a fresh process.
bool in0_own_read_requested() {
    const char* env = std::getenv("TT_SPARSE_MATMUL_EGP_IN0_OWN_READ");
    return env != nullptr && env[0] == '1';
}

}  // namespace

SparseMatmulExpertGroupsProgramFactory::cached_program_t SparseMatmulExpertGroupsProgramFactory::create(
    const ttnn::prim::SparseMatmulParams& operation_attributes,
    const ttnn::prim::SparseMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::Program program{};
    using namespace tt;
    using namespace operations::matmul::utilities;

    TT_FATAL(operation_attributes.expert_groups.has_value(), "EGP factory selected without expert_groups");
    const uint32_t expert_groups = operation_attributes.expert_groups.value();
    TT_FATAL(expert_groups >= 1, "expert_groups must be >= 1, got {}", expert_groups);

    // Same program-config prelude as the legacy sparse factory.
    auto matmul_attributes = ttnn::prim::MatmulParams{
        operation_attributes.program_config,
        /*bcast_batch=*/std::nullopt,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.compute_kernel_config,
        /*untilize_out=*/false,
        operation_attributes.user_core_coord,
        /*user_fused_activation=*/std::nullopt,
        /*user_run_batched=*/false,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        operation_attributes.output_tile,
        operation_attributes.global_cb,
        operation_attributes.sub_device_id};

    auto chosen_program_config = operations::matmul::get_program_config(
        tensor_args.input_tensors.at(0),
        tensor_args.input_tensors.at(1),
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*bias_single_tile_size=*/0,
        matmul_attributes);
    operations::matmul::normalize_program_config(
        chosen_program_config, tensor_args.input_tensors.at(0).device()->compute_with_storage_grid_size());

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& sparsity = tensor_args.input_tensors.at(2);
    const auto& output_tensor = tensor_return_value.at(0);
    TT_FATAL(
        std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config),
        "expert_groups requires a MatmulMultiCoreReuseMultiCast1DProgramConfig");
    auto program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config);
    auto compute_with_storage_grid_size = program_config.allowed_worker_cores.value().bounding_box().grid_size();
    const auto in0_block_w = program_config.in0_block_w;
    const auto out_subblock_h = program_config.out_subblock_h;
    const auto out_subblock_w = program_config.out_subblock_w;
    const auto out_block_h = program_config.out_block_h;
    const auto out_block_w = program_config.out_block_w;
    const auto per_core_M = program_config.per_core_M;
    const auto per_core_N = program_config.per_core_N;
    TT_FATAL(program_config.mcast_in0, "Only mcast_in0 is supported for sparse matmul");

    const auto nnz = operation_attributes.nnz;
    const bool is_input_a_sparse = operation_attributes.is_input_a_sparse;
    TT_FATAL(operation_attributes.is_input_b_sparse, "expert_groups requires is_input_b_sparse=true");
    // bcast_A: in0 is one [M, K] shared by every sparse slot (kept resident in L1); otherwise in0 slot i is
    // block i of a sparse / compact A and is read per owned slot.
    const bool bcast_A = !is_input_a_sparse;

    // Indexed/gather mode (see the legacy factory): the num_active entries of `indices` are the iterated
    // slots, the output is compact, and the in1 kernel's sparsity slot carries the id list.
    const bool use_indices = operation_attributes.use_indices && !tensor_args.optional_input_tensors.empty() &&
                             tensor_args.optional_input_tensors.at(0).has_value();
    uint32_t num_active = 0;
    if (use_indices) {
        num_active = tensor_args.optional_input_tensors.at(0)->logical_volume();
    }

    const auto& ashape = get_matmul_tensor_padded_shape(a, /*transpose=*/false);
    const auto& bshape = get_matmul_tensor_padded_shape(b, /*transpose=*/false);
    const auto in0_tile = get_matmul_tile(a, /*transpose=*/false);
    const auto in1_tile = get_matmul_tile(b, /*transpose=*/false);
    const auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    const auto in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    const auto in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    const auto output_data_format = tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    auto* const device = a.device();

    const auto in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    const auto in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const uint32_t in0_aligned_tile_size = tt::align(in0_single_tile_size, dram_alignment);
    const uint32_t in1_aligned_tile_size = tt::align(in1_single_tile_size, dram_alignment);
    const auto output_single_tile_size = output_tile.get_tile_size(output_data_format);

    auto* const in0_buffer = a.buffer();
    auto* const in1_buffer = b.buffer();
    auto* const sparsity_buffer = sparsity.buffer();
    auto* const out_buffer = output_tensor.buffer();
    const Tensor& in1_sparsity_tensor = use_indices ? tensor_args.optional_input_tensors.at(0).value() : sparsity;
    auto* const in1_sparsity_buffer = in1_sparsity_tensor.buffer();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto batchB = get_batch_size(bshape);
    // A single outer A batch: the A+B-sparse mode has it by construction, the broadcast-A mode by validation
    // (the resident in0 protocol keeps one [per_core_M, Kt] block set for the whole op).
    const uint32_t batchA = is_input_a_sparse ? 1 : get_batch_size(ashape);
    TT_FATAL(batchA == 1, "expert_groups requires a single outer A batch, got {}", batchA);
    TT_FATAL(
        expert_groups <= batchB,
        "expert_groups ({}) must not exceed the number of sparse slots ({})",
        expert_groups,
        batchB);

    const auto Mt = get_M_dim(ashape, in0_tile, /*fuse_batch=*/false);
    const auto Kt = get_K_dim(ashape, in0_tile);
    const auto Nt = get_N_dim(bshape, in1_tile);

    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);

    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_available = num_cores_x * num_cores_y;

    const uint32_t num_blocks_y = ((Mt - 1) / per_core_M) + 1;
    const uint32_t num_blocks_x = ((Nt - 1) / per_core_N) + 1;
    const uint32_t num_blocks_total = num_blocks_y * num_blocks_x;

    // Expert-group grid: G groups x num_blocks_total output blocks. Core i (row-major from start_core) is
    // group i / num_blocks_total and output block i % num_blocks_total.
    const uint32_t num_cores = expert_groups * num_blocks_total;
    TT_FATAL(
        num_cores <= num_cores_available,
        "expert_groups ({}) x output blocks ({}) = {} cores exceeds the {}x{} grid",
        expert_groups,
        num_blocks_total,
        num_cores,
        num_cores_x,
        num_cores_y);
    TT_FATAL(num_cores >= 2, "expert_groups x output blocks must use at least 2 cores, got {}", num_cores);

    using tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids;

    const uint32_t num_blocks = Kt / in0_block_w;
    const bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;
    const auto interm0_data_format = packer_l1_acc_en
                                         ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                         : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);
    const auto interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    const uint32_t in0_block_h = out_block_h;
    const uint32_t in1_block_w = out_block_w;
    const uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    const uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    const uint32_t out_num_blocks_x = in1_num_blocks_x;
    const uint32_t out_num_blocks_y = in0_num_blocks_y;

    // The resident in0 ring is consumed in (h-block, K-block) order, repeated per w-block; that matches the
    // compute kernel's (h-block, w-block, K-block) order only if one of the two block loops is trivial.
    TT_FATAL(
        !bcast_A || out_num_blocks_y == 1 || out_num_blocks_x == 1,
        "expert_groups with a broadcast A requires per_core_M == out_block_h or per_core_N == out_block_w "
        "(per_core_M {} out_block_h {} per_core_N {} out_block_w {})",
        per_core_M,
        out_block_h,
        per_core_N,
        out_block_w);

    const uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    // Broadcast A: the whole [per_core_M, Kt] slice is resident (num_blocks_y x num_blocks blocks, single
    // buffered); sparse A: one double-buffered block, as the legacy factory.
    const uint32_t in0_CB_tiles =
        bcast_A ? out_num_blocks_y * num_blocks * in0_block_tiles
                : in0_block_tiles * ttnn::operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    const uint32_t in0_CB_size = in0_CB_tiles * in0_aligned_tile_size;

    const uint32_t in1_block_tiles = out_block_w * in0_block_w;
    const uint32_t in1_CB_tiles = in1_block_tiles * ttnn::operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    const uint32_t in1_CB_size = in1_CB_tiles * in1_aligned_tile_size;

    const uint32_t out_block_tiles = out_block_h * out_block_w;
    const uint32_t out_CB_size = out_block_tiles * output_single_tile_size;
    const uint32_t interm0_CB_size = out_block_tiles * interm0_single_tile_size;

    const uint32_t sparsity_cb_size = sparsity.buffer()->aligned_page_size();
    const uint32_t in1_sparsity_cb_size = in1_sparsity_buffer->aligned_page_size();

    // L1 budget: fail on the host with the numbers instead of at CB allocation.
    const uint32_t l1_cb_bytes = in0_CB_size + in1_CB_size + out_CB_size +
                                 (interm0_data_format != output_data_format ? interm0_CB_size : 0) + sparsity_cb_size +
                                 in1_sparsity_cb_size;
    const uint32_t l1_budget = tt::tt_metal::hal::get_max_worker_l1_unreserved_size();
    TT_FATAL(
        l1_cb_bytes <= l1_budget,
        "expert_groups: circular buffers need {} B of L1 per core (in0 {} B{}, in1 {} B, out {} B) but only {} B are "
        "available; reduce per_core_M / per_core_N / in0_block_w or use a smaller in0 dtype",
        l1_cb_bytes,
        in0_CB_size,
        bcast_A ? " resident" : " double-buffered",
        in1_CB_size,
        out_CB_size,
        l1_budget);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const CoreCoord start_core = {0, 0};
    CoreRangeSet matmul_core_rect(CoreRange(
        start_core,
        CoreCoord(
            start_core.x + compute_with_storage_grid_size.x - 1, start_core.y + compute_with_storage_grid_size.y - 1)));

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores, matmul_core_rect, row_major);
    const CoreRange all_cores_bounding_box = all_cores.bounding_box();
    // in0 is multicast to the bounding box, so every core in it must run the kernels.
    TT_FATAL(
        num_cores == all_cores_bounding_box.size(),
        "expert_groups ({}) x output blocks ({}) = {} cores must fill an exact rectangle of the {}x{} grid (bounding "
        "box holds {} cores); adjust the grid or the number of groups",
        expert_groups,
        num_blocks_total,
        num_cores,
        num_cores_x,
        num_cores_y,
        all_cores_bounding_box.size());

    // Mcast semaphores (used by the mcast-once resident load only; created in every mode for one arg layout).
    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    const CoreCoord top_left_core = all_cores_bounding_box.start_coord;
    const CoreCoord bottom_right_core = all_cores_bounding_box.end_coord;
    const auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    const auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);
    const auto sender_core_physical = device->worker_core_from_logical_core(start_core);

    // Compute / in1 loop count: every scanned slot (E) or indexed entry; validity comes from the in0 kernel's
    // mailbox writes in every mode (never a static nnz loop count).
    const uint32_t num_batch_compute = use_indices ? num_active : batchB;
    constexpr bool get_batch_from_reader = true;
    const uint32_t expected_nnz = use_indices ? 0 : nnz.value_or(0);
    // Compact output ([1, nnz, M, N]) detection exactly as the legacy factory / device op.
    const bool compact_output =
        nnz.has_value() &&
        output_tensor.logical_shape() == ttnn::Shape{1U, nnz.value(), a.logical_shape()[-2], b.logical_shape()[-1]};

    const uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    const uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    const auto& a_shape_logical = get_matmul_tensor_logical_shape(a, /*transpose=*/false);
    const auto in0_last_ktile_w = a_shape_logical[-1] % in0_tile.get_width();

    const uint32_t in0_tensor_stride_w = 1;
    const uint32_t in0_tensor_stride_h = Kt;
    const uint32_t in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const uint32_t in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;

    const uint32_t in1_tensor_stride_w = 1;
    const uint32_t in1_tensor_stride_h = Nt;
    const uint32_t in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
    const uint32_t in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;

    // in0 EGP kernel compile-time args (layout documented in reader_bmm_tile_layout_in0_expert_groups.cpp).
    std::vector<uint32_t> in0_compile_time_args = {
        (std::uint32_t)in0_tensor_stride_w,
        (std::uint32_t)in0_tensor_stride_h,
        (std::uint32_t)in0_tensor_next_block_stride,
        (std::uint32_t)in0_tensor_next_h_dim_block_stride,
        (std::uint32_t)in0_block_w,
        (std::uint32_t)in0_block_h,
        (std::uint32_t)in0_block_num_tiles,
        (std::uint32_t)in0_last_ktile_w,
        (std::uint32_t)num_blocks,
        (std::uint32_t)out_num_blocks_x,
        (std::uint32_t)out_num_blocks_y,
        (std::uint32_t)in0_mcast_sender_semaphore_id,
        (std::uint32_t)in0_mcast_receiver_semaphore_id,
        (std::uint32_t)num_cores - 1,  // in0_mcast_num_dests
        (std::uint32_t)Mt * Kt,        // MtKt
        (std::uint32_t)batchB,
        (std::uint32_t)sparsity_buffer->aligned_page_size(),  // sparsity_pagesize
        (std::uint32_t)bcast_A,
        (std::uint32_t)expected_nnz,
        (std::uint32_t)expert_groups,
    };
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(in0_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*sparsity_buffer).append_to(in0_compile_time_args);

    // in1 sender/writer compile-time args: identical to the legacy sparse factory (the shared kernel).
    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        // READER
        (std::uint32_t)in1_tensor_stride_w,
        (std::uint32_t)in1_tensor_stride_h,
        (std::uint32_t)in1_tensor_next_block_stride,
        (std::uint32_t)in1_tensor_next_w_dim_block_stride,
        (std::uint32_t)in1_block_w,                               // in1_block_w
        (std::uint32_t)in0_block_w,                               // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,                 // in1_block_num_tiles
        (std::uint32_t)num_blocks,                                // num_blocks
        (std::uint32_t)out_num_blocks_x,                          // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,                          // out_num_blocks_y
        (std::uint32_t)0,                                         // in1 mcast sender semaphore (unused, SKIP_MCAST)
        (std::uint32_t)0,                                         // in1 mcast receiver semaphore (unused, SKIP_MCAST)
        (std::uint32_t)0,                                         // in1_mcast_num_dests
        (std::uint32_t)0,                                         // in1_mcast_num_cores
        (std::uint32_t)Kt * Nt,                                   // KtNt
        (std::uint32_t)batchA,                                    // batch (outer A loop count)
        (std::uint32_t)true,                                      // bcast_B
        (std::uint32_t)batchB,                                    // batchB
        (std::uint32_t)in1_sparsity_buffer->aligned_page_size(),  // sparsity_pagesize
        // WRITER
        (std::uint32_t)1,                    // out_tensor_stride_w
        (std::uint32_t)Nt,                   // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h
        (std::uint32_t)out_block_w,          // out_tensor_next_w_dim_block_stride
        (std::uint32_t)out_block_h * Nt,     // out_tensor_next_h_dim_block_stride
        (std::uint32_t)out_subblock_w,
        (std::uint32_t)out_subblock_h,
        (std::uint32_t)(out_subblock_w * out_subblock_h),
        (std::uint32_t)Mt * Nt,  // MtNt
        (std::uint32_t)0,        // in3_tensor_stride_w (bias placeholder)
        (std::uint32_t)false,    // fuse_op
        (std::uint32_t)false,    // fuse_op_reduce_scatter
        (std::uint32_t)compact_output,
    };
    tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(in1_sender_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*in1_sparsity_buffer).append_to(in1_sender_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(in1_sender_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in1_sender_writer_compile_time_args);  // placeholder for bias

    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_defines;
    std::map<std::string, std::string> mm_kernel_in1_defines;

    mm_kernel_defines["FUSE_ACTIVATION"] = "0";
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(),
        num_cores,
        mm_kernel_defines,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config));

    const bool in0_mcast_once = bcast_A && !in0_own_read_requested();
    if (bcast_A) {
        mm_kernel_in0_defines["IN0_RESIDENT"] = "1";
    }
    if (in0_mcast_once) {
        mm_kernel_in0_defines["IN0_MCAST_ONCE"] = "1";
    }
    mm_kernel_in1_defines["SKIP_MCAST"] = "1";
    mm_kernel_in1_defines["EXPERT_GROUPS"] = std::to_string(expert_groups);
    mm_kernel_in1_defines["EXPERT_GROUPS_NNZ"] = std::to_string(expected_nnz);

    const tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    const tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    auto in0_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_expert_groups.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in0_noc,
            .compile_args = in0_compile_time_args,
            .defines = mm_kernel_in0_defines,
            .named_compile_args = {
                {"cb_in0", tt::CBIndex::c_0},
                {"cb_sparsity", tt::CBIndex::c_6},
                {"num_active", num_active},
            }});

    auto in1_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in1_noc,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_kernel_in1_defines,
            .named_compile_args = {
                {"cb_in1", tt::CBIndex::c_1},
                {"cb_bias", tt::CBIndex::c_3},
                {"cb_out", tt::CBIndex::c_4},
                {"cb_sparsity", tt::CBIndex::c_7},
                {"num_active", num_active},
            }});

    // Compute kernel: unchanged kernel, legacy compile args except batch (= scanned slots) and
    // get_batch_from_reader (always on).
    const uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    const uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
    const uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    const uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    const uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_per_core_w,
        num_blocks,
        out_num_blocks_x,
        out_num_blocks_y,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        num_batch_compute,      // batch
        out_block_tiles,        // out_block_num_tiles
        false,                  // untilize_out
        get_batch_from_reader,  // get_batch_from_reader
        false,                  // in0_transpose_tile
    };

    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines,
            .named_compile_args = {
                {"cb_in0", tt::CBIndex::c_0},
                {"cb_in1", tt::CBIndex::c_1},
                {"cb_bias", tt::CBIndex::c_3},
                {"cb_out", tt::CBIndex::c_4},
                {"cb_intermed0", tt::CBIndex::c_5},
                {"cb_in0_transposed", tt::CBIndex::c_10},
            }});

    // Circular buffers (same indices / formats / page sizes as the legacy factory).
    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_aligned_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    const uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_aligned_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);
    tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);

    const uint32_t output_cb_index = tt::CBIndex::c_4;
    const uint32_t interm0_cb_index = tt::CBIndex::c_5;
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});
    if (interm0_data_format != output_data_format) {
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{{output_cb_index, output_data_format}};
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile);
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{{interm0_cb_index, interm0_data_format}};
        tt_metal::CircularBufferConfig interm0_cb_config =
            tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                .set_tile_dims(interm0_cb_index, output_tile);
        tt_metal::CreateCircularBuffer(program, all_cores, interm0_cb_config);
    } else {
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_page_size(interm0_cb_index, interm0_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile)
                               .set_tile_dims(interm0_cb_index, output_tile);
    }
    tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    const uint32_t sparsity_cb_index0 = tt::CBIndex::c_6;
    const uint32_t sparsity_cb_index1 = tt::CBIndex::c_7;
    tt_metal::CircularBufferConfig sparsity_cb_config0 =
        tt_metal::CircularBufferConfig(
            sparsity_cb_size, {{sparsity_cb_index0, tt::tt_metal::datatype_to_dataformat_converter(sparsity.dtype())}})
            .set_page_size(sparsity_cb_index0, sparsity_cb_size);
    tt_metal::CircularBufferConfig sparsity_cb_config1 =
        tt_metal::CircularBufferConfig(
            in1_sparsity_cb_size,
            {{sparsity_cb_index1, tt::tt_metal::datatype_to_dataformat_converter(in1_sparsity_tensor.dtype())}})
            .set_page_size(sparsity_cb_index1, in1_sparsity_cb_size);
    tt_metal::CreateCircularBuffer(program, all_cores, sparsity_cb_config0);
    tt_metal::CreateCircularBuffer(program, all_cores, sparsity_cb_config1);

    // Last-column padding parameters (no split on height), as the legacy factory.
    const uint32_t last_per_core_N = Nt % per_core_N == 0 ? per_core_N : Nt % per_core_N;
    const uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    const uint32_t last_out_num_blocks_w = ((last_per_core_N - 1) / out_block_w) + 1;
    const uint32_t last_block_num_nonzero_subblocks_w = ((last_out_block_w - 1) / out_subblock_w) + 1;
    const uint32_t last_subblock_of_last_block_w =
        last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
    const uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    const uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        const uint32_t group_id = i / num_blocks_total;
        const uint32_t block_id = i % num_blocks_total;
        const uint32_t output_idx_x = block_id % num_blocks_x;
        const uint32_t output_idx_y = block_id / num_blocks_x;

        // in0 EGP kernel runtime args: [0] in0 addr, [1] in0 start tile id, [2..5] mcast rectangle,
        // [6..7] sender NoC x/y, [8] last_block_h, [9] sparsity addr, [10] group_id, [11] is_sender.
        std::vector<uint32_t> in0_args = {
            (std::uint32_t)in0_buffer->address(),
            (std::uint32_t)Kt * per_core_M * output_idx_y,  // in0_tensor_start_tile_id
            (std::uint32_t)start_core_noc.x,
            (std::uint32_t)start_core_noc.y,
            (std::uint32_t)end_core_noc.x,
            (std::uint32_t)end_core_noc.y,
            (std::uint32_t)sender_core_physical.x,
            (std::uint32_t)sender_core_physical.y,
            (std::uint32_t)out_block_h,  // last_block_h
            (std::uint32_t)sparsity_buffer->address(),
            (std::uint32_t)group_id,
            (std::uint32_t)(core == start_core ? 1 : 0),
        };
        tt_metal::SetRuntimeArgs(program, in0_kernel_id, core, in0_args);

        // in1 sender/writer runtime args: the legacy 26 words with [21] = group_id (read under EXPERT_GROUPS).
        std::vector<uint32_t> in1_args = {
            (std::uint32_t)in1_buffer->address(),
            (std::uint32_t)per_core_N * output_idx_x,  // in1_tensor_start_tile_id
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)in1_sparsity_buffer->address(),  // sparsity_addr (id list in indexed mode)
            (std::uint32_t)out_buffer->address(),
            ((std::uint32_t)output_idx_x * per_core_N) + (output_idx_y * per_core_M * Nt)  // out_tensor_start_tile_id
        };
        if (output_idx_x == num_blocks_x - 1) {
            in1_args.push_back(last_out_block_w);
            in1_args.push_back(out_block_h / out_subblock_h);
            in1_args.push_back(out_subblock_h);
            in1_args.push_back(0);
            in1_args.push_back(out_block_w / out_subblock_w);
            in1_args.push_back(last_block_num_nonzero_subblocks_w);
            in1_args.push_back(last_subblock_of_last_block_w);
            in1_args.push_back(last_block_padded_subblock_tiles_addr_skip);
            in1_args.push_back(last_block_padded_block_tiles_w_skip);
        } else {
            in1_args.push_back(out_block_w);
            in1_args.push_back(out_block_h / out_subblock_h);
            in1_args.push_back(out_subblock_h);
            in1_args.push_back(0);
            in1_args.push_back(out_block_w / out_subblock_w);
            in1_args.push_back(out_block_w / out_subblock_w);
            in1_args.push_back(out_subblock_w);
            in1_args.push_back(0);
            in1_args.push_back(0);
        }
        in1_args.push_back(0);  // [18] bias placeholder
        in1_args.push_back(0);  // [19] bias placeholder
        in1_args.push_back(output_idx_x == num_blocks_x - 1 ? last_out_num_blocks_w : out_num_blocks_x);  // [20]
        in1_args.push_back(group_id);  // [21] expert group
        in1_args.push_back(0);
        in1_args.push_back(0);
        in1_args.push_back(0);
        in1_args.push_back(0);
        tt_metal::SetRuntimeArgs(program, in1_kernel_id, core, in1_args);
    }

    log_debug(
        LogOp,
        "sparse_matmul EGP: G {} x {} blocks = {} cores ({}x{} grid), in0 {} ({} B), in1 CB {} B, batch {}, "
        "expected_nnz {}, compact {}, indexed {}",
        expert_groups,
        num_blocks_total,
        num_cores,
        num_cores_x,
        num_cores_y,
        bcast_A ? (in0_mcast_once ? "resident/mcast-once" : "resident/own-read") : "per-slot",
        in0_CB_size,
        in1_CB_size,
        num_batch_compute,
        expected_nnz,
        compact_output,
        use_indices);

    auto shared_vars = SparseMatmulExpertGroupsProgramFactory::shared_variables_t{
        in0_kernel_id, in1_kernel_id, cores, num_cores, expert_groups};
    return {std::move(program), std::move(shared_vars)};
}

void SparseMatmulExpertGroupsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::SparseMatmulParams& operation_attributes,
    const ttnn::prim::SparseMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    auto* src_buffer_a = tensor_args.input_tensors.at(0).buffer();
    auto* src_buffer_b = tensor_args.input_tensors.at(1).buffer();
    auto* sparsity_buffer = tensor_args.input_tensors.at(2).buffer();
    auto* dst_buffer = tensor_return_value.at(0).buffer();

    const bool use_indices = operation_attributes.use_indices && !tensor_args.optional_input_tensors.empty() &&
                             tensor_args.optional_input_tensors.at(0).has_value();
    auto* in1_sparsity_buffer = use_indices ? tensor_args.optional_input_tensors.at(0)->buffer() : sparsity_buffer;

    auto& in0_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.in0_kernel_id);
    auto& in1_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.in1_kernel_id);
    for (uint32_t i = 0; i < shared_vars.num_cores; ++i) {
        const auto& core = shared_vars.cores[i];
        auto& in0_runtime_args = in0_runtime_args_by_core[core.x][core.y];
        in0_runtime_args[0] = src_buffer_a->address();
        in0_runtime_args[9] = sparsity_buffer->address();

        auto& in1_runtime_args = in1_runtime_args_by_core[core.x][core.y];
        in1_runtime_args[0] = src_buffer_b->address();
        in1_runtime_args[6] = in1_sparsity_buffer->address();
        in1_runtime_args[7] = dst_buffer->address();
    }
}

}  // namespace ttnn::prim
