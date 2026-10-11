// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 Phase F F5 (MOTIF3_CCL_FUSE=all): the race-free ring all-gather of a tile-aligned TILE tensor as ONE program
// (ttnn's all_broadcast + concat, the "safe" gather of tt/ccl.py, in one): every chip reads its tile range of the input,
// waits for the start barrier (every peer runs this program, so its output buffer is live), multicasts the tiles along
// the ring straight into the peers' outputs at their concatenated positions (forward hops 1..fwd_range, backward hops
// 1..bwd_range: every peer exactly once; 2 tiles per scatter packet when they fit), writes them into its own output,
// then one multicast atomic inc of its source slot's counter on the SAME routes (Flush: the writes before it have
// landed; no alternate-route completion race), and waits until every peer's tiles have landed here (monotonic
// counters: absolute targets invocation + 1, never reset). Pure data movement: bitwise the concat.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"

#include <cstdint>
#include <algorithm>

#include "cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_common.hpp"

#ifndef MOTIF_CCLAR_EXP
#define MOTIF_CCLAR_EXP 0
#endif
// MOTIF_CCLAR_EXP (timing experiments only: MOTIF3_CCLAR_EXP): 1 = no start barrier (UNSAFE), 2 = no gather sends /
// waits, 4 = no fabric at all (local copy only).

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t num_devices = get_compile_time_arg_val(1);
    constexpr uint32_t run_pages = get_compile_time_arg_val(2);     // input pages per gather-dim block
    constexpr uint32_t stride_pages = get_compile_time_arg_val(3);  // output pages per gather-dim block row
    constexpr bool pairs = get_compile_time_arg_val(4) != 0;        // 2 tiles fit one fabric packet
    constexpr uint32_t cb_id = get_compile_time_arg_val(5);
    constexpr auto input_args = TensorAccessorArgs<6>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t num_dests = num_devices - 1;

    size_t ai = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t output_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t device_idx = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_start = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_count = get_arg_val<uint32_t>(ai++);
    const uint32_t gen_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t init_sem = get_arg_val<uint32_t>(ai++);
    const uint32_t ag_sem = get_arg_val<uint32_t>(ai++);
    const uint8_t peer_core_x = get_arg_val<uint32_t>(ai++);
    const uint8_t peer_core_y = get_arg_val<uint32_t>(ai++);
    const uint32_t num_connections = get_arg_val<uint32_t>(ai++);
    const uint8_t fwd_range = static_cast<uint8_t>(get_arg_val<uint32_t>(ai++));
    const uint8_t bwd_range = static_cast<uint8_t>(get_arg_val<uint32_t>(ai++));
    const size_t ag_wait_sems = ai;
    ai += num_dests;
    size_t arg_for_fab = ai;

    auto input_acc = TensorAccessor(input_args, input_addr);
    auto output_acc = TensorAccessor(output_args, output_addr);
    Noc noc;
    CircularBuffer cb(cb_id);

    auto* gen_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gen_addr);
    const uint32_t invocation = *gen_ptr;

    if constexpr (MOTIF_CCLAR_EXP & 4) {
        cb.reserve_back(tile_count);
        const uint32_t l1x = cb.get_write_ptr();
        for (uint32_t i = 0; i < tile_count; ++i) {
            noc.async_read(input_acc, CoreLocalMem<uint32_t>(l1x + i * tile_bytes), tile_bytes, {.page_id = tile_start + i}, {}, {});
        }
        noc.async_read_barrier();
        for (uint32_t i = 0; i < tile_count; ++i) {
            noc.async_write(CoreLocalMem<uint32_t>(l1x + i * tile_bytes), output_acc, tile_bytes, {},
                            {.page_id = device_idx * run_pages + ((tile_start + i) / run_pages) * stride_pages + (tile_start + i) % run_pages}, {});
        }
        noc.async_write_barrier();
        return;
    }
    auto fabric_connection = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
        arg_for_fab, num_connections);

    // this core's input tiles -> L1 (overlaps the connection handshake)
    cb.reserve_back(tile_count);
    const uint32_t l1 = cb.get_write_ptr();
    for (uint32_t i = 0; i < tile_count; ++i) {
        noc.async_read(input_acc, CoreLocalMem<uint32_t>(l1 + i * tile_bytes), tile_bytes, {.page_id = tile_start + i}, {}, {});
    }

    uint8_t starts[2] = {1, 1};
    uint8_t ranges[2] = {fwd_range, bwd_range};
    const auto r_scatter = PacketHeaderPool::allocate_header_n(num_connections);
    const auto r_single = PacketHeaderPool::allocate_header_n(num_connections);
    const auto r_inc = PacketHeaderPool::allocate_header_n(num_connections);
    fabric_connection.open_finish();
    if constexpr (pairs) {
        fabric_api::fabric_multicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            fabric_connection,
            r_scatter,
            starts,
            ranges,
            tt::tt_fabric::NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(tile_bytes)}),
            tile_bytes * 2);
    }
    fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        fabric_connection, r_single, starts, ranges, nullptr, tile_bytes);
    fabric_api::fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection, r_inc, starts, ranges, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1u});

    // start barrier: one multicast inc per direction (every peer gets exactly one from us)
    noc.async_read_barrier();
    if constexpr (!(MOTIF_CCLAR_EXP & 1)) {
        const uint64_t init_noc_addr = safe_get_noc_addr(peer_core_x, peer_core_y, init_sem, 0);
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection, r_inc, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_noc_addr, 0});
        noc.async_writes_flushed();
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_sem), (invocation + 1) * num_dests);
    }

    const uint32_t slice_base = device_idx * run_pages;
    auto out_page = [&](uint32_t i) -> uint32_t {
        return slice_base + (i / run_pages) * stride_pages + (i % run_pages);
    };
    for (uint32_t i = 0; i < ((MOTIF_CCLAR_EXP & 2) ? 0u : tile_count);) {
        const uint64_t a0 = tt::tt_fabric::addrgen_detail::get_noc_address(output_acc, out_page(tile_start + i), 0);
        if (pairs && i + 1 < tile_count) {
            const uint64_t a1 =
                tt::tt_fabric::addrgen_detail::get_noc_address(output_acc, out_page(tile_start + i + 1), 0);
            fabric_api::fabric_multicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                fabric_connection,
                r_scatter,
                l1 + i * tile_bytes,
                tt::tt_fabric::NocUnicastScatterCommandHeader({a0, a1}),
                tile_bytes * 2);
            i += 2;
        } else {
            fabric_api::fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                fabric_connection, r_single, l1 + i * tile_bytes, tt::tt_fabric::NocUnicastCommandHeader{a0});
            i += 1;
        }
        noc.async_writes_flushed();
    }
    for (uint32_t i = 0; i < tile_count; ++i) {
        noc.async_write(
            CoreLocalMem<uint32_t>(l1 + i * tile_bytes), output_acc, tile_bytes, {}, {.page_id = out_page(tile_start + i)}, {});
    }
    if constexpr (!(MOTIF_CCLAR_EXP & 2)) {
        const uint64_t ag_noc_addr = safe_get_noc_addr(peer_core_x, peer_core_y, ag_sem, 0);
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection, r_inc, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{ag_noc_addr, 0});
        noc.async_writes_flushed();
    }

    for (uint32_t s = 0; s < ((MOTIF_CCLAR_EXP & 2) ? 0u : num_dests); ++s) {
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(ag_wait_sems + s));
        noc_semaphore_wait_min(sem, invocation + 1);
    }
    noc_semaphore_set(gen_ptr, invocation + 1);

    fabric_connection.close_start();
    noc.async_write_barrier();
    noc.async_atomic_barrier();
    fabric_connection.close_finish();
}
