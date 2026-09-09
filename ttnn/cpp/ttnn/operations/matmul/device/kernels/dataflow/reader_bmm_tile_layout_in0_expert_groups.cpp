// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// in0 reader of the sparse matmul EXPERT-GROUP PARALLELISM (EGP) program factory
// (device/sparse/factory/sparse_matmul_expert_groups_program_factory.cpp). Runs on EVERY core of the
// G x (output blocks) grid; the role of the multicast sender is a runtime argument.
//
// Compared with the legacy in0 sender / receiver pair:
//   * the validity of every sparsity slot is decided LOCALLY: the non-zero slots are numbered by their
//     running rank r in scan order and this core computes slot r iff r % expert_groups == group_id
//     (indexed mode: entry i iff i % expert_groups == group_id). The decision reaches the compute kernel
//     through the same mailbox writes the legacy kernels use, so there is no per-slot multicast round trip;
//   * IN0_RESIDENT (broadcast A, is_input_b_sparse only): the [per_core_M, Kt] in0 blocks are loaded ONCE into
//     a CB that holds exactly num_blocks_h_dim x num_blocks_inner_dim blocks -- with IN0_MCAST_ONCE the sender
//     reads them and multicasts each block to the whole grid (one handshake per block per op), otherwise every
//     core reads them itself -- and every later owned expert re-uses them with the dense "reuse_in0_in_CB"
//     reserve/push pointer dance (no data movement);
//   * without IN0_RESIDENT (is_input_a_sparse, e.g. the MoE down projection): each owned slot's in0 blocks are
//     read from DRAM by this core into the double-buffered CB, no multicast at all.
// The bytes handed to compute per (slot, block) are identical to the legacy kernels', so the result is
// bit-identical; only the (slot, block) -> core assignment changes.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

void kernel_main() {
    uint32_t rt_args_idx = 0;
    // in0 tensor args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    // in0 mcast args (the whole grid's bounding box in NoC coordinates; used by the sender only)
    [[maybe_unused]] const uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    [[maybe_unused]] const uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    [[maybe_unused]] const uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    [[maybe_unused]] const uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);
    // the sender core in NoC coordinates (used by the receivers only)
    [[maybe_unused]] const uint32_t in0_mcast_sender_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    [[maybe_unused]] const uint32_t in0_mcast_sender_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    // padding args
    const uint32_t last_block_h = get_arg_val<uint32_t>(rt_args_idx++);
    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);
    // expert-group args
    const uint32_t group_id = get_arg_val<uint32_t>(rt_args_idx++);
    [[maybe_unused]] const bool is_sender = get_arg_val<uint32_t>(rt_args_idx++) != 0;

    // COMPILE TIME ARGS
    // in0 tensor args
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in0_tensor_next_inner_dim_block_stride = get_compile_time_arg_val(2);
    constexpr uint32_t in0_tensor_next_h_dim_block_stride = get_compile_time_arg_val(3);
    // in0 block args
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(5);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t in0_last_ktile_w = get_compile_time_arg_val(7);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(8);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(9);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(10);
    // in0 mcast args (indices 11 and 12 are the semaphore ids, consumed below)
    [[maybe_unused]] constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(13);
    // batch args
    constexpr uint32_t MtKt = get_compile_time_arg_val(14);
    // sparsity args
    constexpr uint32_t batchB = get_compile_time_arg_val(15);
    constexpr uint32_t sparsity_pagesize = get_compile_time_arg_val(16);
    // True when in0 is broadcast over the sparse slots ([1, 1, M, K], only B sparse); false when in0 slot i is
    // block i of a sparse / compact A.
    constexpr bool bcast_A = (bool)get_compile_time_arg_val(17);
    // Caller-supplied nnz (0 when the count is inferred). Ranks >= expected_nnz are never computed (the in1
    // writer applies the same bound) and the total is asserted below.
    constexpr uint32_t expected_nnz = get_compile_time_arg_val(18);
    constexpr uint32_t expert_groups = get_compile_time_arg_val(19);
    static_assert(expert_groups >= 1, "expert_groups must be >= 1");

    constexpr auto in0_args = TensorAccessorArgs<20>();
    constexpr auto sparsity_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    // Indexed/gather mode: iterate the num_active entries of the caller's `indices` operand instead of
    // scanning the batchB sparsity slots; every entry is valid (0 = not indexed).
    constexpr uint32_t num_active = get_named_compile_time_arg_val("num_active");
    constexpr bool use_indices = num_active > 0;
    constexpr uint32_t batch_loop_lim = use_indices ? num_active : batchB;

    constexpr uint32_t dfb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(dfb_id_in0);
    // Interleaved tiles are read at the DRAM-aligned stride into DRAM-aligned CB pages (see the factory).
    constexpr uint32_t in0_aligned_tile_size_bytes =
        (in0_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
    [[maybe_unused]] constexpr uint32_t in0_block_size_bytes = in0_block_num_tiles * in0_aligned_tile_size_bytes;

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);
    [[maybe_unused]] Semaphore<> sender_sem(get_compile_time_arg_val(11));
    [[maybe_unused]] Semaphore<> receiver_sem(get_compile_time_arg_val(12));

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);

    // sparsity page (scan modes only; the indexed mode never reads the mask)
    constexpr uint32_t dfb_id_sparsity = get_named_compile_time_arg_val("cb_sparsity");
    DataflowBuffer dfb_sparsity(dfb_id_sparsity);
    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr);
    uint32_t l1_write_addr_sparsity = 0;
    if constexpr (!use_indices) {
        dfb_sparsity.reserve_back(1);
        l1_write_addr_sparsity = dfb_sparsity.get_write_ptr();
        noc.async_read(s_sparsity, dfb_sparsity, sparsity_pagesize, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
    }

    // Reads one [in0_block_h, in0_block_w] block of in0 (tile ids from block_start_tile_id) into the slot the
    // caller has just reserved in dfb_in0; the caller issues the read barrier. Identical tile walk (including
    // the K-padding of the last tile) to the legacy in0 sender.
    auto read_in0_block = [&](uint32_t block_start_tile_id, uint32_t bh, uint32_t block) {
        uint32_t in0_write_offset = 0;
        uint32_t in0_tensor_row_start_tile_id = block_start_tile_id;
        for (uint32_t h = 0; h < in0_block_h; ++h) {
            uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
            for (uint32_t w = 0; w < in0_block_w; ++w) {
                if (bh < num_blocks_h_dim - 1 || h < last_block_h) {
                    noc.async_read(
                        s0,
                        dfb_in0,
                        in0_single_tile_size_bytes,
                        {.page_id = in0_tensor_tile_id},
                        {.offset_bytes = in0_write_offset});
                }
                // Zero out padded regions for the very last tile
                if constexpr (in0_last_ktile_w > 0) {
                    if ((block == num_blocks_inner_dim - 1) && (w == in0_block_w - 1)) {
                        noc.async_read_barrier();
                        constexpr DataFormat in0_data_format = get_dataformat(dfb_id_in0);
                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(dfb_in0.get_write_ptr() + in0_write_offset);
                    }
                }
                in0_write_offset += in0_aligned_tile_size_bytes;
                in0_tensor_tile_id += in0_tensor_stride_w;
            }
            in0_tensor_row_start_tile_id += in0_tensor_stride_h;
        }
    };

    // Blocks already pushed into the resident CB that the first owned expert consumes without a fake push.
    [[maybe_unused]] uint32_t pending_real_blocks = 0;

#ifdef IN0_RESIDENT
    // One-shot load of the resident in0: num_blocks_h_dim x num_blocks_inner_dim blocks, exactly the CB's
    // capacity (the w-block loop re-uses them, see the ownership loop below).
#ifdef IN0_MCAST_ONCE
    if (is_sender) {
        // Local VALID value, multicast to the receivers' flag after each block has landed.
        receiver_sem.set(VALID);
    }
#endif  // IN0_MCAST_ONCE
    {
        uint32_t in0_tensor_current_h_dim_block_tile_id = in0_tensor_start_tile_id;
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            uint32_t in0_tensor_current_inner_dim_block_start_tile_id = in0_tensor_current_h_dim_block_tile_id;
            for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                dfb_in0.reserve_back(in0_block_num_tiles);
#ifdef IN0_MCAST_ONCE
                if (is_sender) {
                    const uint32_t in0_start_address = dfb_in0.get_write_ptr();
                    read_in0_block(in0_tensor_current_inner_dim_block_start_tile_id, bh, block);
                    noc.async_read_barrier();

                    // Wait until every receiver has reserved this block's slot, then multicast the data
                    // and the VALID flag (the legacy sender's per-block protocol, issued once per op).
                    sender_sem.wait(in0_mcast_num_dests);
                    sender_sem.set(0);
                    MulticastEndpoint mcast_dst;
                    noc.async_write_multicast(
                        CoreLocalMem<uint32_t>(in0_start_address),
                        mcast_dst,
                        in0_block_size_bytes,
                        in0_mcast_num_dests,
                        {},
                        {.noc_x_start = in0_mcast_dest_noc_start_x,
                         .noc_y_start = in0_mcast_dest_noc_start_y,
                         .noc_x_end = in0_mcast_dest_noc_end_x,
                         .noc_y_end = in0_mcast_dest_noc_end_y,
                         .addr = in0_start_address},
                        true);
#ifdef ARCH_BLACKHOLE
                    // On Blackhole the flush is needed because NoC latency is higher than L1 <-> RISCV
                    // latency which means data could be changed before write is issued.
                    noc.async_writes_flushed();
#endif  // ARCH_BLACKHOLE
                    receiver_sem.set_multicast(
                        noc,
                        in0_mcast_dest_noc_start_x,
                        in0_mcast_dest_noc_start_y,
                        in0_mcast_dest_noc_end_x,
                        in0_mcast_dest_noc_end_y,
                        in0_mcast_num_dests);
                } else {
                    receiver_sem.set(INVALID);
                    sender_sem.up(noc, in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, 1);
                    receiver_sem.wait(VALID);
                }
#else   // IN0_MCAST_ONCE
                read_in0_block(in0_tensor_current_inner_dim_block_start_tile_id, bh, block);
                noc.async_read_barrier();
#endif  // IN0_MCAST_ONCE
                dfb_in0.push_back(in0_block_num_tiles);
                in0_tensor_current_inner_dim_block_start_tile_id += in0_tensor_next_inner_dim_block_stride;
            }
            in0_tensor_current_h_dim_block_tile_id += in0_tensor_next_h_dim_block_stride;
        }
    }
    pending_real_blocks = num_blocks_h_dim * num_blocks_inner_dim;
#endif  // IN0_RESIDENT

    // Ownership scan. Every slot costs one mailbox write per TRISC (the legacy per-slot cost minus the grid
    // round trip); owned slots additionally push num_blocks_h_dim x num_blocks_w_dim x num_blocks_inner_dim
    // in0 blocks for compute, in the compute kernel's consumption order.
    [[maybe_unused]] uint32_t rank = 0;
    for (uint32_t i = 0; i < batch_loop_lim; ++i) {
        bool is_mine = false;
        if constexpr (use_indices) {
            is_mine = (i % expert_groups) == group_id;
        } else {
            const bool is_batch_valid =
                (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_sparsity))[i] != 0;
            const uint32_t r = rank;
            if (is_batch_valid) {
                ++rank;
            }
            is_mine = is_batch_valid && (r % expert_groups) == group_id && (expected_nnz == 0 || r < expected_nnz);
        }

        // The compute kernel reads one validity word per batch iteration on each of its three threads.
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, is_mine);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, is_mine);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, is_mine);
        if (!is_mine) {
            continue;
        }

        [[maybe_unused]] uint32_t in0_tensor_current_h_dim_block_tile_id =
            bcast_A ? in0_tensor_start_tile_id : in0_tensor_start_tile_id + i * MtKt;
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                [[maybe_unused]] uint32_t in0_tensor_current_inner_dim_block_start_tile_id =
                    in0_tensor_current_h_dim_block_tile_id;
                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
#ifdef IN0_RESIDENT
                    // The resident blocks are consumed in exactly the order they were pushed; once the first
                    // owned expert has used the real pushes, every further block is a reserve/push pair that
                    // re-presents the same L1 tiles to compute (the dense reuse_in0_in_CB mechanism). The
                    // reserve blocks until compute has popped the block that occupied the slot.
                    if (pending_real_blocks > 0) {
                        --pending_real_blocks;
                    } else {
                        dfb_in0.reserve_back(in0_block_num_tiles);
                        dfb_in0.push_back(in0_block_num_tiles);
                    }
#else   // IN0_RESIDENT
                    dfb_in0.reserve_back(in0_block_num_tiles);
                    read_in0_block(in0_tensor_current_inner_dim_block_start_tile_id, bh, block);
                    noc.async_read_barrier();
                    dfb_in0.push_back(in0_block_num_tiles);
                    in0_tensor_current_inner_dim_block_start_tile_id += in0_tensor_next_inner_dim_block_stride;
#endif  // IN0_RESIDENT
                }
            }
            in0_tensor_current_h_dim_block_tile_id += in0_tensor_next_h_dim_block_stride;
        }
    }

    // Exact-nnz contract (scan modes with a caller-supplied nnz): count_nonzero(sparsity) must equal nnz.
    // A mismatch can no longer deadlock the device (validity is local), but the output would silently hold
    // fewer / different slots than announced -- fail loudly under watcher instead.
    if constexpr (!use_indices && expected_nnz > 0) {
        ASSERT(rank == expected_nnz);
    }

    noc.async_write_barrier();
    noc.async_atomic_barrier();

    // For completeness, empty the sparsity CB if it was reserved earlier
    if constexpr (!use_indices) {
        dfb_sparsity.push_back(1);
        dfb_sparsity.wait_front(1);
        dfb_sparsity.pop_front(1);
    }
}
