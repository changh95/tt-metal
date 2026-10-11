// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 Phase F F5 (MOTIF3_CCL_FUSE=ar): the decode all-reduce as ONE program = the direct reduce-scatter
// (ttnn.experimental.reduce_scatter_minimal_direct, whose reader and compute kernels are used unchanged from
// ccl_ar/reader.cpp and ccl_ar/compute.cpp) + an all-gather phase in this writer.
//
// This is ttnn's reduce_scatter_minimal_direct_writer.cpp (tt-metal c1879f2b, same protocol, same send order, same
// start barrier) with two changes:
//
//   1. The reduced slice is written into the FULL all-reduce output (same shape as the input) at the pages the slice
//      occupies there (the reader's (run, stride) walk of slice `device_idx`), i.e. where ttnn.concat of the
//      gathered reduce-scatter outputs puts it.
//   2. All-gather phase: the same tiles are unicast to every peer's output (same address on every chip: the mesh
//      allocator is lockstep), farthest peer first, one tile per packet, the LAST packet to a peer a fused write +
//      atomic inc of that peer's per-SOURCE gather counter (flush: the payload lands before the inc). Data and
//      counter travel the SAME unicast route, so the even-ring alternate-route completion race of ttnn's multicast
//      all_gather (tt/ccl.py docstring) cannot occur. The writer then waits until every peer's tiles have landed in
//      this chip's output (absolute monotonic targets invocation + 1, per source: never reset, as the reduce
//      arrivals). Peers only write after the start barrier, i.e. once this chip runs THIS program (its output
//      buffer is live), so no write can hit a buffer an earlier program on this chip still uses.
//
// The reduction itself (block order: own slice, then the remote sources in ascending device order; the compute
// kernel's pairwise DST fold) is the reduce-scatter's, so the result is bitwise the reduce-scatter + all-gather +
// concat chain's.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"  // tt_fabric::addrgen_detail::get_noc_address
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"

#include <cstdint>
#include <algorithm>
#include <type_traits>

#include "cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_common.hpp"

using address_t = uint32_t;

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(1);
    constexpr uint32_t chunks_per_slice = get_compile_time_arg_val(2);
    [[maybe_unused]] constexpr uint32_t pages_per_slice = get_compile_time_arg_val(3);
    constexpr uint32_t num_devices = get_compile_time_arg_val(4);
    constexpr uint32_t interm_tiles_per_packet = get_compile_time_arg_val(5);
    constexpr uint32_t cb_send_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(7);
    constexpr bool arrivals_in_cb = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t half_stride_tiles = get_compile_time_arg_val(9);
    constexpr bool needs_init_sync = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t slice_run_pages = get_compile_time_arg_val(11);  // F5: the output walk of our slice
    constexpr uint32_t stride_pages = get_compile_time_arg_val(12);
    constexpr bool gather = get_compile_time_arg_val(13) != 0;  // F5: the all-gather phase (else: RS only)
    constexpr auto staging_args = TensorAccessorArgs<14>();
    constexpr auto output_args = TensorAccessorArgs<staging_args.next_compile_time_args_offset()>();

    constexpr uint32_t num_dests = num_devices - 1;

    size_t ai = 0;
    const address_t staging_addr = get_arg_val<address_t>(ai++);
    const address_t output_addr = get_arg_val<address_t>(ai++);
    const uint32_t device_idx = get_arg_val<uint32_t>(ai++);
    const uint32_t chunk_start = get_arg_val<uint32_t>(ai++);
    const uint32_t chunk_count = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_start = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_count = get_arg_val<uint32_t>(ai++);
    const address_t gen_addr = get_arg_val<uint32_t>(ai++);
    const address_t arrival_sem = get_arg_val<uint32_t>(ai++);
    const uint8_t peer_core_x = get_arg_val<uint32_t>(ai++);
    const uint8_t peer_core_y = get_arg_val<uint32_t>(ai++);
    const uint32_t num_connections = get_arg_val<uint32_t>(ai++);
    const address_t init_sem = get_arg_val<uint32_t>(ai++);
    const uint8_t fwd_mcast_range = static_cast<uint8_t>(get_arg_val<uint32_t>(ai++));
    const uint8_t bwd_mcast_range = static_cast<uint8_t>(get_arg_val<uint32_t>(ai++));
    const address_t ag_sem = get_arg_val<uint32_t>(ai++);  // F5: our source slot's gather counter on every peer
    const size_t ag_wait_sems = ai;                          // F5: the gather counters of the remote sources
    ai += num_dests;
    const size_t dest_conn = ai;
    ai += num_dests;
    const size_t dest_hops = ai;
    ai += num_dests;
    const size_t dest_block = ai;
    ai += num_dests;
    const size_t dest_chip = ai;
    ai += num_dests;
    const size_t dest_mesh = ai;
    ai += num_dests;
    size_t arg_for_fab = ai;

    auto output_acc = TensorAccessor(output_args, output_addr);

    Noc noc;
    CircularBuffer cb_send(cb_send_id);
    CircularBuffer cb_out(cb_out_id);

    auto* gen_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gen_addr);
    const uint32_t invocation = *gen_ptr;
    const uint32_t staging_half = (invocation & 1u) * (num_devices * chunks_per_slice);
    const uint32_t dst_page_base = staging_half + device_idx * chunks_per_slice + chunk_start;
    const uint32_t half_off_bytes = (invocation & 1u) * half_stride_tiles * tile_bytes;
    constexpr uint32_t chunk_bytes = tile_granularity * tile_bytes;

    auto fabric_connection = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
        arg_for_fab, num_connections);

    auto data_pkt = PacketHeaderPool::allocate_header(1);
    auto fused_pkt = PacketHeaderPool::allocate_header(1);

    const uint64_t arrival_noc_addr = safe_get_noc_addr(peer_core_x, peer_core_y, arrival_sem, 0);

    auto set_route_2d = [&](volatile PACKET_HEADER_TYPE* hdr, uint32_t dst) {
        if constexpr (std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PACKET_HEADER_TYPE>) {
            fabric_set_unicast_route(
                reinterpret_cast<volatile tt::tt_fabric::HybridMeshPacketHeader*>(hdr),
                static_cast<uint16_t>(get_arg_val<uint32_t>(dest_chip + dst)),
                static_cast<uint16_t>(get_arg_val<uint32_t>(dest_mesh + dst)));
        }
    };

    fabric_connection.open_finish();

    // ---- F5 gather routes (multicast along the ring, as ttnn's all_broadcast: forward hops 1..fwd_mcast_range and
    // backward hops 1..bwd_mcast_range, together every peer exactly once) ----
    uint8_t starts[2] = {1, 1};
    uint8_t ranges[2] = {fwd_mcast_range, bwd_mcast_range};
    const auto ag_scatter = PacketHeaderPool::allocate_header_n(num_connections);
    const auto ag_single = PacketHeaderPool::allocate_header_n(num_connections);
    const auto ag_inc = PacketHeaderPool::allocate_header_n(num_connections);
    if constexpr (gather) {
        fabric_api::fabric_multicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            fabric_connection,
            ag_scatter,
            starts,
            ranges,
            tt::tt_fabric::NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(tile_bytes)}),
            tile_bytes * 2);
        fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            fabric_connection, ag_single, starts, ranges, nullptr, tile_bytes);
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            fabric_connection, ag_inc, starts, ranges, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1u});
    }

#ifndef MOTIF_CCLAR_EXP
#define MOTIF_CCLAR_EXP 0
#endif
    // MOTIF_CCLAR_EXP (timing experiments only, never values: MOTIF3_CCLAR_EXP): 1 = no start barrier (UNSAFE),
    // 2 = no gather sends / waits (the output holds only this chip's slice).
    if constexpr (gather && needs_init_sync && !(MOTIF_CCLAR_EXP & 1)) {
        // Start barrier as one multicast atomic inc per direction (every peer gets exactly one from us).
        const uint64_t init_noc_addr = safe_get_noc_addr(peer_core_x, peer_core_y, init_sem, 0);
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection, ag_inc, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_noc_addr, 0});
        noc.async_writes_flushed();
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_sem), (invocation + 1) * num_dests);
    } else if constexpr (needs_init_sync && !(MOTIF_CCLAR_EXP & 1)) {
        auto init_pkt = PacketHeaderPool::allocate_header(1);
        const uint64_t init_noc_addr = safe_get_noc_addr(peer_core_x, peer_core_y, init_sem, 0);
        if constexpr (std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PACKET_HEADER_TYPE>) {
            for (uint32_t dst = 0; dst < num_dests; ++dst) {
                auto* sender = &fabric_connection.get(get_arg_val<uint32_t>(dest_conn + dst)).sender;
                fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
                    UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val |
                    UnicastAtomicIncUpdateMask::Flush>(
                    init_pkt,
                    static_cast<uint8_t>(get_arg_val<uint32_t>(dest_hops + dst)),
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_noc_addr, 1u});
                set_route_2d(init_pkt, dst);
                fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::None>(
                    sender, init_pkt);
                noc.async_writes_flushed();
            }
        } else {
            if (fwd_mcast_range > 0) {
                fabric_api::fabric_multicast_noc_unicast_atomic_inc(
                    &fabric_connection.get(0).sender,
                    init_pkt,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_noc_addr, 1u},
                    /*start_distance=*/1,
                    fwd_mcast_range);
                noc.async_writes_flushed();
            }
            if (bwd_mcast_range > 0) {
                fabric_api::fabric_multicast_noc_unicast_atomic_inc(
                    &fabric_connection.get(1).sender,
                    init_pkt,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_noc_addr, 1u},
                    /*start_distance=*/1,
                    bwd_mcast_range);
                noc.async_writes_flushed();
            }
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_sem), (invocation + 1) * num_dests);
    }

    // ---- reduce-scatter sends (unchanged) ----
    for (uint32_t dst = 0; dst < num_dests; ++dst) {
        const uint8_t hops = static_cast<uint8_t>(get_arg_val<uint32_t>(dest_hops + dst));
        auto* sender = &fabric_connection.get(get_arg_val<uint32_t>(dest_conn + dst)).sender;
        const uint32_t block_base =
            arrivals_in_cb ? staging_addr + half_off_bytes + get_arg_val<uint32_t>(dest_block + dst) * chunk_bytes : 0u;

        fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(data_pkt, hops);
        fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
            UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush>(
            fused_pkt, hops, tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0u, 0u, 1u}, /*packet_size_bytes=*/0);
        set_route_2d(data_pkt, dst);
        set_route_2d(fused_pkt, dst);

        uint32_t tiles_done = 0;
        for (uint32_t k = 0; k < chunk_count; ++k) {
            const uint32_t tiles_in_chunk = std::min(tile_granularity, tile_count - tiles_done);
            cb_send.wait_front(tile_granularity);
            const uint32_t rd = cb_send.get_read_ptr();

            for (uint32_t t = 0; t < tiles_in_chunk; t += interm_tiles_per_packet) {
                const uint32_t tiles_in_pkt = std::min(tiles_in_chunk - t, interm_tiles_per_packet);
                const uint16_t payload = tiles_in_pkt * tile_bytes;
                uint64_t dst_noc;
                if constexpr (arrivals_in_cb) {
                    dst_noc = safe_get_noc_addr(
                        peer_core_x, peer_core_y, block_base + k * num_devices * chunk_bytes + t * tile_bytes, 0);
                } else {
                    auto staging_acc = TensorAccessor(staging_args, staging_addr);
                    dst_noc =
                        tt::tt_fabric::addrgen_detail::get_noc_address(staging_acc, dst_page_base + k, t * tile_bytes);
                }
                const bool last_packet = (k == chunk_count - 1) && (t + interm_tiles_per_packet >= tiles_in_chunk);
                if (last_packet) {
                    fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                        UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
                        UnicastFusedAtomicIncUpdateMask::PayloadSize>(
                        sender,
                        fused_pkt,
                        rd + t * tile_bytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc, arrival_noc_addr, 1u},
                        payload);
                } else {
                    fabric_api::fabric_unicast_noc_unicast_write_with_state<
                        UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                        sender, data_pkt, rd + t * tile_bytes, tt::tt_fabric::NocUnicastCommandHeader{dst_noc}, payload);
                }
                noc.async_writes_flushed();
            }

            cb_send.pop_front(tile_granularity);
            tiles_done += tiles_in_chunk;
        }
    }

    // Output page of tile i of our slice (the reader's input walk of slice device_idx; the AR output has the input's
    // shape and page order).
    const uint32_t slice_base = device_idx * slice_run_pages;
    auto out_page = [&](uint32_t i) -> uint32_t {
        return slice_base + (i / slice_run_pages) * stride_pages + (i % slice_run_pages);
    };

    // ---- reduced slice -> local output + gather, chunk by chunk. The CB holds every chunk of this core (sized by the
    // host) and nothing is popped until the end, so tile i of this core sits at rd0 + i * tile_bytes. ----
    uint32_t rd0 = 0;
    {
        uint32_t tiles_done = 0;
        for (uint32_t k = 0; k < chunk_count; ++k) {
            const uint32_t tiles_in_chunk = std::min(tile_granularity, tile_count - tiles_done);
            cb_out.wait_front((k + 1) * tile_granularity);
            if (k == 0) {
                rd0 = cb_out.get_read_ptr();
            }
            const uint32_t i0 = k * tile_granularity;
            if constexpr (gather && !(MOTIF_CCLAR_EXP & 2)) {
                for (uint32_t t = 0; t < tiles_in_chunk; t += 2) {
                    const uint32_t i = i0 + t;
                    const uint64_t a0 =
                        tt::tt_fabric::addrgen_detail::get_noc_address(output_acc, out_page(tile_start + i), 0);
                    if (t + 1 < tiles_in_chunk) {
                        const uint64_t a1 =
                            tt::tt_fabric::addrgen_detail::get_noc_address(output_acc, out_page(tile_start + i + 1), 0);
                        fabric_api::fabric_multicast_noc_scatter_write_with_state<
                            UnicastScatterWriteUpdateMask::DstAddrs>(
                            fabric_connection,
                            ag_scatter,
                            rd0 + i * tile_bytes,
                            tt::tt_fabric::NocUnicastScatterCommandHeader({a0, a1}),
                            tile_bytes * 2);
                    } else {
                        fabric_api::fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            fabric_connection,
                            ag_single,
                            rd0 + i * tile_bytes,
                            tt::tt_fabric::NocUnicastCommandHeader{a0});
                    }
                    noc.async_writes_flushed();
                }
            }
            for (uint32_t t = 0; t < tiles_in_chunk; ++t) {
                const uint32_t i = i0 + t;
                noc.async_write(
                    CoreLocalMem<uint32_t>(rd0 + i * tile_bytes),
                    output_acc,
                    tile_bytes,
                    {},
                    {.page_id = out_page(tile_start + i)},
                    {});
            }
            tiles_done += tiles_in_chunk;
        }
    }
    if constexpr (gather && !(MOTIF_CCLAR_EXP & 2)) {
        // One atomic inc of our source slot's gather counter on every peer, on the SAME routes as the data (Flush: the
        // writes before it have landed).
        const uint64_t ag_noc_addr = safe_get_noc_addr(peer_core_x, peer_core_y, ag_sem, 0);
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection, ag_inc, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{ag_noc_addr, 0});
    }
    noc.async_writes_flushed();
    cb_out.pop_front(chunk_count * tile_granularity);

    if constexpr (gather && !(MOTIF_CCLAR_EXP & 2)) {
        // Every peer's slice (this core's tile range of it) has landed in our output.
        for (uint32_t s = 0; s < num_dests; ++s) {
            auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(ag_wait_sems + s));
            noc_semaphore_wait_min(sem, invocation + 1);
        }
    }
    noc_semaphore_set(gen_ptr, invocation + 1);

    fabric_connection.close_start();
    noc.async_write_barrier();
    noc.async_atomic_barrier();
    fabric_connection.close_finish();
}
