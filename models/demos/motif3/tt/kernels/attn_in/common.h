// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode attention input chain (Phase F, F1; tt/kernels/attn_in.py): CB / semaphore ids and the work
// layout shared by the dataflow and compute kernels (host mirror: attn_in.py).
#pragma once

#include <cstdint>

namespace motif_ain {

// ---- circular buffers (ids; every CB that is written remotely has one address on every core) --------------------
constexpr uint32_t CB_CQN = 0;     // fp32 32: cq_n (QN: norm output; QB cores: multicast receive, matmul in0)
constexpr uint32_t CB_WA = 1;      // bf16 16: QB weights, K half 0 (RISC 0)
constexpr uint32_t CB_WB = 2;      // bf16 16: QB weights, K half 1 (RISC 1)
constexpr uint32_t CB_QOUT = 3;    // bf16 1: QB matmul result (compute's copy: gate -> sigmoid, pe -> RoPE)
constexpr uint32_t CB_QSEND = 4;   // bf16 1: QB matmul result (RISC 1's copy: nope -> UK cores, pe -> partner)
constexpr uint32_t CB_GOUT = 5;    // bf16 1: sigmoid(gate tile)
constexpr uint32_t CB_NOPE = 6;    // bf16 4: UK in0 (q_nope of the head; written by the 4 nope producers)
constexpr uint32_t CB_WUK = 7;     // bf16 8: UK weights (2 units x 4 k tiles)
constexpr uint32_t CB_UKOUT = 8;   // bf16 2: UK results
constexpr uint32_t CB_PEPART = 9;  // bf16 1: the partner's pre-RoPE pe tile
constexpr uint32_t CB_COS = 10;    // bf16 2
constexpr uint32_t CB_SIN = 11;    // bf16 2
constexpr uint32_t CB_SCAL = 12;   // bf16 1: RoPE scalar tile (-1 at element 0)
constexpr uint32_t CB_ROT = 13;    // bf16 1: RoPE rotated interm
constexpr uint32_t CB_CI = 14;     // bf16 1: RoPE cos interm
constexpr uint32_t CB_SI = 15;     // bf16 1: RoPE sin interm
constexpr uint32_t CB_ROPE = 16;   // bf16 1: RoPE output (q pe tile)
constexpr uint32_t CB_QMLA = 17;   // bf16 L: owner assembly of q_mla column tiles (one per lane)
constexpr uint32_t CB_KVROW = 18;  // bf16 18: [n | rope(kpe)] (KN)
constexpr uint32_t CB_QX = 19;     // fp32 32: cq (QN)
constexpr uint32_t CB_XMM2 = 20;   // fp32 32: x^2 (QN / KN norm)
constexpr uint32_t CB_EX2 = 21;    // fp32 1
constexpr uint32_t CB_EX2PE = 22;  // fp32 1
constexpr uint32_t CB_RSCAL = 23;  // bf16 1: reduce scaler (1.0 in row 0 of every face)
constexpr uint32_t CB_EPS = 24;    // bf16 1: eps in column 0 of faces 0 / 2
constexpr uint32_t CB_KIN = 25;    // bf16 16: c_raw (KN)
constexpr uint32_t CB_ROTIN = 26;  // bf16 2: KN RoPE rotated-input stream
constexpr uint32_t CB_KPEX = 27;   // bf16 2: KN RoPE input stream
constexpr uint32_t CB_LAM = 28;    // bf16 2: lam tiles (KN)

// ---- semaphores ------------------------------------------------------------------------------------------------
constexpr uint32_t SEM_CQN = 0;   // QB cores: cq_n blocks landed (set by QN, value = blocks)
constexpr uint32_t SEM_NOPE = 1;  // UK cores: nope tiles landed (incremented by the 4 producers)
constexpr uint32_t SEM_PE = 2;    // pe cores: the partner's tile landed
constexpr uint32_t SEM_OWN = 3;   // owners: head tiles landed (10 per column)
constexpr uint32_t SEM_LOC = 4;   // QN: RISC 1's half of cq landed (local)

// ---- geometry --------------------------------------------------------------------------------------------------
constexpr uint32_t H = 10;         // q heads per chip
constexpr uint32_t HT = 6;         // q tiles per head: 4 nope + 2 pe
constexpr uint32_t NOPE_T = 4;
constexpr uint32_t KQ = 32;        // K tiles of q_b / gate (q_lora_rank 1024)
constexpr uint32_t KH = KQ / 2;
constexpr uint32_t NQ = H * HT;    // 60 q_b columns
constexpr uint32_t NG = 32;        // gate columns
constexpr uint32_t NQB = NQ + NG;  // 92 QB units, unit u on core u
constexpr uint32_t UKN = 16;       // W_UK output tiles per head (kv_lora_rank 512)
constexpr uint32_t UK_PER = 2;     // UK units per core
constexpr uint32_t UK_CPH = UKN / UK_PER;  // 8 UK cores per head: core h * 8 + i owns units 2i, 2i + 1
constexpr uint32_t NUK = H * UK_CPH;       // 80 UK cores
constexpr uint32_t NCOL = UKN + 2;         // 18 q_mla column tiles (576 / 32)
constexpr uint32_t OWN0 = 92;              // owner of q_mla column col: core OWN0 + col
constexpr uint32_t KVT = 18;               // kv_row tiles
constexpr uint32_t KVIN = 20;              // kvl tiles: c_raw 16 | kpe 2 | lam 2
constexpr uint32_t QN_BLOCKS = KQ / 4;     // cq_n multicast blocks (the norm's block size 4)
constexpr uint32_t BF16_TILE = 2048;
constexpr uint32_t FP32_TILE = 4096;

}  // namespace motif_ain
