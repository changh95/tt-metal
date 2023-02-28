#include <algorithm>

#include "ll_buda/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include "ll_buda/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace tt {

namespace ll_buda {

Tensor eltwise_unary_multi_core(const Tensor &a, UnaryOpType::Enum op_type) {
    ll_buda::Program *program = new ll_buda::Program();


    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to eltwise unary needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to eltwise unary needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    ll_buda::InterleavedDramBuffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume() / TILE_HW;

    // InterleavedDramBuffer stores buffers across multiple dram banks but reader kernel only needs the location of the first one
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates().at(0);

    ll_buda::Device *device = a.device();

    auto logical_grid_size = device->logical_grid_size();
    int32_t num_cores_x = logical_grid_size.x;
    int32_t num_cores_y = logical_grid_size.y;
    auto num_cores = std::min(num_tiles, num_cores_x * num_cores_y);
    std::vector<int32_t> num_tiles_per_core(num_cores, num_tiles / num_cores);
    for(uint32_t i = 0; i < num_tiles % num_cores; i++){
        num_tiles_per_core[i]++;
    }

    // This should allocate a DRAM buffer on the device
    ll_buda::Tensor output = ll_buda::Tensor(a.shape(), a.dtype(), tt::ll_buda::Layout::TILE, device);

    ll_buda::InterleavedDramBuffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    // InterleavedDramBuffer stores buffers across multiple dram banks but writer kernel only needs the location of the first one
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates().at(0);

    std::vector<ll_buda::DataMovementKernel *> unary_reader_kernels;
    std::vector<ll_buda::DataMovementKernel *> unary_writer_kernels;
    for (uint32_t i = 0; i < num_cores; i++){
        tt_xy_pair core = {i / num_cores_y, i % num_cores_y};
        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = 2;
        auto cb_src0 = ll_buda::CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src0_cb_addr,
            DataFormat::Float16_b
        );

        uint32_t src1_cb_index = 1;
        uint32_t src1_cb_addr = 300 * 1024;
        auto cb_src1 = ll_buda::CreateCircularBuffer(
            program,
            device,
            src1_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src1_cb_addr,
            DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = 2;
        auto cb_output = ll_buda::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            DataFormat::Float16_b
        );

        ll_buda::DataMovementKernel *unary_reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_unary_8bank_start_id.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);
        unary_reader_kernels.push_back(unary_reader_kernel);

        ll_buda::DataMovementKernel *unary_writer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_unary_8bank_start_id.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);
        unary_writer_kernels.push_back(unary_writer_kernel);

        void *hlk_args = new eltwise_unary::hlk_args_t{
            .per_core_block_cnt = num_tiles_per_core[i],
            .per_core_block_size = 1
        };
        ll_buda::ComputeKernelArgs *eltwise_unary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(eltwise_unary::hlk_args_t));

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_unary_kernel = ll_buda::CreateComputeKernel(
            program,
            "kernels/compute/eltwise_sfpu.cpp",
            core,
            eltwise_unary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        set_compute_kernel_defines(eltwise_unary_kernel, op_type);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool skip_hlkc = false;
    ll_buda::CompileProgram(device, program, skip_hlkc);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    ll_buda::ConfigureDeviceWithProgram(device, program);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; num_tiles_written+=num_tiles_per_core[i], i++){
        tt_xy_pair core = {i / num_cores_y, i % num_cores_y};
        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_reader_kernels[i],
            core,
            {src0_dram_buffer->address(),
            uint32_t(dram_src0_noc_xy.x),
            uint32_t(dram_src0_noc_xy.y),
            uint32_t(num_tiles_per_core[i]),
            num_tiles_written }
        );

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernels[i],
            core,
            {dst_dram_buffer->address(),
            uint32_t(dram_dst_noc_xy.x),
            uint32_t(dram_dst_noc_xy.y),
            uint32_t(num_tiles_per_core[i]),
            num_tiles_written }
        );
    }

    ll_buda::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace ll_buda

}  // namespace tt
