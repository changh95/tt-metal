// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused device-sampler per-chip stage (Phase F, F4; tt/kernels/sampler_local.py): CB / semaphore ids, roles
// and tile helpers shared by the dataflow and compute kernels (host mirror: sampler_local.py).
#pragma once

#include <cstdint>

namespace motif_sl {

// ---- roles ---------------------------------------------------------------------------------------------------------
constexpr uint32_t ROLE_PROD = 0;  // logical rows 0 .. GY-2: TPP vocab tiles each (q = y GX + x)
constexpr uint32_t ROLE_ACC = 1;   // (j, GY-1), j < 8: face f = j % 4 of the row sum of exp(a) (j < 4) / exp(d)
constexpr uint32_t ROLE_AGG = 2;   // (8, GY-1): inputs multicast, Gumbel cross-core max / argmin / raw, u, q
constexpr uint32_t NACC = 8;
constexpr uint32_t AGG_X = 8;

// ---- circular buffers (the first eight span every core of the program: one L1 address everywhere) --------------------
constexpr uint32_t CB_EA = 0;    // fp32 TPP: exp(a) tiles of a producer (read remotely by the ACC cores j < 4)
constexpr uint32_t CB_ED = 1;    // fp32 TPP: exp(d) tiles (ACC cores j >= 4)
constexpr uint32_t CB_PM = 2;    // fp32 NPT: AGG, the producers' local score maxima, producer q in row q % 32 of tile q / 32
constexpr uint32_t CB_PL = 3;    // fp32 NPT: AGG, the producers' lowest local id attaining the max
constexpr uint32_t CB_PD = 4;    // fp32 NPT: AGG, the producers' raw x - M at the max
constexpr uint32_t CB_MS = 5;    // fp32 1: the per-row max score broadcast along the row (AGG multicasts it to PROD)
constexpr uint32_t CB_CF = 6;    // fp32 2: M, 1 / T broadcast along the rows (AGG builds / reads, multicasts to PROD)
constexpr uint32_t CB_CI = 7;    // int32 2: c1, c2 broadcast along the rows (host-written rows 32..95 of the counters)
// producers
constexpr uint32_t CB_X = 8;     // bf16 TPP: logits tiles
constexpr uint32_t CB_GK = 9;    // int32 TPP: Gumbel token keys K_v
constexpr uint32_t CB_IO = 10;   // fp32 TPP: local ids 0 .. 6879 (the sampler's iota_vc)
constexpr uint32_t CB_D = 11;    // fp32 TPP: d = x - M
constexpr uint32_t CB_A = 12;    // fp32 TPP: a = d / T
constexpr uint32_t CB_S = 13;    // fp32 TPP: Gumbel scores s
constexpr uint32_t CB_W = 14;    // fp32 1: local max of s (column 0)
constexpr uint32_t CB_W2 = 15;   // fp32 2: local argmin id / raw at the global max (column 0)
constexpr uint32_t CB_STG = 16;  // fp32 1: 128 B column staging (3 slots); AGG: the -0 face-row pattern
constexpr uint32_t CB_T = 29;    // fp32 TPP: log(-F) = log(E), the Gumbel score's RHS (the subtract's activations)
// accumulators (identical CB lists on the 8 ACC cores: CB_AF has one address on all of them)
constexpr uint32_t CB_RING = 17; // fp32 RING: face f of 4 consecutive e tiles per page, vocab order
constexpr uint32_t CB_S2 = 18;   // fp32 1: this core's accumulated face (face 0 of the packed tile)
constexpr uint32_t CB_AF = 19;   // fp32 1: master (f = 0): the four accumulated faces = the full fold tile
constexpr uint32_t CB_S2B = 20;  // fp32 1: master: the reduced row sum (column 0)
constexpr uint32_t CB_SF = 21;   // fp32 3: ACC master: s2 in column 0 (zeros elsewhere); AGG: m, gid, dw likewise
constexpr uint32_t CB_E0 = 22;   // fp32 1: e0 (1 in column 0, else 0) [ACC master, AGG]
constexpr uint32_t CB_OUT = 23;  // fp32 3: ACC master: the part tile; AGG: the three q tiles
// aggregator
constexpr uint32_t CB_GR = 24;   // fp32 3: reduced m, li, dw (row 0)
constexpr uint32_t CB_U = 25;    // fp32 1: u (column 0)
constexpr uint32_t CB_OFF = 26;  // fp32 1: the raw chip_off tile
constexpr uint32_t CB_RAW = 27;  // fp32 1: the raw M tile
constexpr uint32_t CB_RAWI = 28; // int32 1: the raw counter tile (c1 column 0, c2 column 1)
constexpr uint32_t CB_TL = 30;   // uint32 1: timeline stamps (experiments only)

// ---- semaphores (every core) ---------------------------------------------------------------------------------------
constexpr uint32_t SEM_E = 0;    // ACC: producers whose e tiles are packed (NP)
constexpr uint32_t SEM_PM = 1;   // AGG: producers whose local max row landed (NP)
constexpr uint32_t SEM_PL = 2;   // AGG: producers whose li / dw rows landed (NP)
constexpr uint32_t SEM_M = 3;    // PROD: the max-score broadcast tile landed (1)
constexpr uint32_t SEM_RAW = 4;  // PROD: the M, 1 / T broadcasts landed (1; reset by the PROD reader)
constexpr uint32_t SEM_AF = 5;   // ACC master: accumulated faces landed (4)
constexpr uint32_t SEM_L0 = 6;   // AGG local: RISC 1 -> RISC 0, the raw M tile is in L1
constexpr uint32_t SEM_L1 = 7;   // AGG local: RISC 0 -> RISC 1, rows 16..31 of the M broadcast are built
constexpr uint32_t SEM_L2 = 8;   // AGG local: RISC 0 -> RISC 1, m (row 0 of CB_GR page 0) is ready
constexpr uint32_t SEM_L3 = 9;   // AGG local: RISC 1 -> RISC 0, rows 16..31 of the m broadcast are built
constexpr uint32_t SEM_CI = 10;  // PROD: the c1, c2 broadcasts landed (1; reset by the PROD reader)

// ---- constants -----------------------------------------------------------------------------------------------------
constexpr uint32_t FP32_TILE = 4096;
constexpr uint32_t FACE = 1024;           // fp32 face bytes
constexpr uint32_t BF16_TILE = 2048;
constexpr uint32_t ONE_F = 0x3F800000u;   // 1.0f
constexpr uint32_t NEG_INF = 0xFF800000u; // identity rows of the max folds
constexpr uint32_t POS_INF = 0x7F800000u; // identity rows of the min fold

// word offset of element (r, c) of a 32 x 32 tile (four 16 x 16 faces, row major inside a face)
inline constexpr uint32_t tw(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

}  // namespace motif_sl
