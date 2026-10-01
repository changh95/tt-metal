// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 router_fp32: optional wall-clock stamps (tests only; compiled out when the TIMING CT arg is 0).
// Every RISC of a core writes its stamps (low 32 bits of the core's wall clock, AICLK cycles) into word slots of the
// core's CB_TIME page; the last writer of the core copies the page to row <core> of a DRAM uint32 tensor.
//
// Worker slots (RISC-V side times: Tensix-side waits do not stall the RISC-V, so math / pack stamps are push times):
// 0 unpack start, 1 unpack: chunk-0 weights arrived, 2 unpack: chunk-0 x arrived, 3 unpack: chunk-3 x arrived,
// 4 math start, 5 math: all SFPU instructions pushed (~ SFPU done: the Tensix FIFO is 32 deep), 6 / 7 pack (push
// times), 8 writer start, 9 writer: partials packed, 10 writer: reduce-scatter NoC writes done, 11 writer: semaphores
// done, 12 reader start, 13 reader: all reads done, 14 reader: all 32 pieces received, 15 writer: final piece written.
#pragma once

#include <cstdint>

inline __attribute__((always_inline)) void motif_router_stamp(uint32_t base_addr, uint32_t slot) {
    reinterpret_cast<volatile uint32_t*>(base_addr)[slot] =
        *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("fence" ::: "memory");
}
