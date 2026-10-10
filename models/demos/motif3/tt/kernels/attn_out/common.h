// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode attention output chain (Phase F, F2; tt/kernels/attn_out.py): CB / semaphore ids and the
// work layout shared by the dataflow and compute kernels (host mirror: attn_out.py).
#pragma once

#include <cstdint>

namespace motif_aout {

// ---- circular buffers ------------------------------------------------------------------------------------------
// remotely written (one address on every core: allocated on the whole grid first)
constexpr uint32_t CB_SIG = 0;     // bf16 1: u signal tile (h, t), written by UV core (h, t)            [CMB]
constexpr uint32_t CB_NOISE = 1;   // bf16 1: u noise tile (8 + h / 4, t), written by that UV core        [CMB]
constexpr uint32_t CB_DGIN = 2;    // bf16 32: the wo input dg, tile j written by CMB core j (FUSED)      [WO]
constexpr uint32_t CB_IN0 = 3;     // bf16 16: o_heads tiles k 0..15 of head h; core (h, t) builds k 4t..4t+3 [UV]
// UV cores (W_UV bmm, one output tile each)
constexpr uint32_t CB_STAGE = 4;   // bf16 9: 64 B-aligned staging of the o_lat row reads (RISC r: 8 KB at 8 KB r)
constexpr uint32_t CB_WA = 5;      // bf16 8: W_UV tiles (k, t), k 0..7 (RISC 0)
constexpr uint32_t CB_WB = 6;      // bf16 8: k 8..15 (RISC 1)
constexpr uint32_t CB_PART = 7;    // fp32 1: the reuse bmm's K-block partial (spill / TF32 reload)
constexpr uint32_t CB_U = 8;       // bf16 1: u tile (h, t)
// CMB cores (D1 combine + lam expansion, fp32 DEST off)
constexpr uint32_t CB_LAM = 9;     // bf16 2: lam [L, 64]
constexpr uint32_t CB_E = 10;      // bf16 2: E tiles (0, j), (1, j)
constexpr uint32_t CB_VPRE = 11;   // bf16 1: lam @ E (tile j)
constexpr uint32_t CB_V = 12;      // bf16 1: sigmoid(lam @ E)
constexpr uint32_t CB_G = 13;      // bf16 1: g tile j                   (UnpackToDestFp32: binary_ng SFPU operand)
constexpr uint32_t CB_ACT = 14;    // bf16 1: active tile j              (UnpackToDestFp32)
constexpr uint32_t CB_D = 15;      // bf16 1: addcmul result             (UnpackToDestFp32)
constexpr uint32_t CB_DG = 16;     // bf16 1: d * g                      (UnpackToDestFp32)
constexpr uint32_t CB_OUTC = 17;   // bf16 1: where(active, dg, 0) = the wo input tile j
constexpr uint32_t CB_VDBG = 18;   // bf16 1: debug copy of v
// WO cores (FUSED)
constexpr uint32_t CB_WOA = 19;    // bf16 32: wo tiles (k, n0 + i), k 0..15, i 0..1 ([k][i]): chunks 0..3 (RISC 0)
constexpr uint32_t CB_WOB = 20;    // bf16 16: k 16..23: chunks 4, 5 (UV cores: RISC 0; WO-only cores: RISC 1)
constexpr uint32_t CB_WOUT = 21;   // bf16 2: wo output tiles n0, n0 + 1
constexpr uint32_t CB_TL = 22;     // bf16 1: timeline stamps (experiments; RISC r at 64 B r)
constexpr uint32_t CB_WOC = 23;    // bf16 16: k 24..31: chunks 6, 7 (RISC 1; UV cores after their chain work)

// ---- semaphores --------------------------------------------------------------------------------------------------
constexpr uint32_t SEM_SIG = 0;    // CMB: the signal tile landed
constexpr uint32_t SEM_NOISE = 1;  // CMB: the noise tile landed
constexpr uint32_t SEM_DG = 2;     // WO: dg tiles landed (32)
constexpr uint32_t SEM_IN0 = 3;    // UV: the head's other 3 quarters landed (3)
constexpr uint32_t SEM_L1 = 4;     // UV: level-1 rows of this core's quarter landed (10 cores x 2 RISCs = 20)

// ---- geometry ----------------------------------------------------------------------------------------------------
constexpr uint32_t H = 10;         // virtual heads per chip: 8 signal (Sg) + 2 noise (G)
constexpr uint32_t SG = 8;
constexpr uint32_t HPG = 4;        // signal heads per noise group
constexpr uint32_t TPH = 4;        // u tiles per head (v_head_dim 128)
constexpr uint32_t KUV = 16;       // K tiles of the W_UV bmm (kv_lora_rank 512)
constexpr uint32_t KH = KUV / 2;   // per RISC
constexpr uint32_t UV_IBW = 4;     // the stock reuse config's in0_block_w (attention ATTN_DECODE_BMM["w_uv"])
constexpr uint32_t KQ4 = KUV / TPH;  // o_heads tiles per quarter (4): core (h, t) owns k 4t .. 4t + 3
constexpr uint32_t ROWB = H * 32;  // o_lat rows 0..9 of one face: 320 B (one DRAM read serves the 10 heads)
constexpr uint32_t NUV = H * TPH;  // 40 UV units: core (x, y), x < 8, y < 5 -> unit 8 y + x = (h, t) = (u / 4, u % 4)
constexpr uint32_t NCMB = SG * TPH;  // 32 combine units: core (8 + x, y), x < 4, y < 8 -> tile j = 4 y + x
constexpr uint32_t UV_GX = 8;
constexpr uint32_t CMB_X0 = 8;
constexpr uint32_t NLAMT = 2;      // lam tiles (64 columns)
constexpr uint32_t KWO = 32;       // K tiles of wo (1024)
constexpr uint32_t KWOH = KWO / 2;
constexpr uint32_t NWO_PER = 2;    // wo output tiles per WO core (64 cores x 2 = 128 = 4096 / 32)
constexpr uint32_t NWO_T = 128;    // wo output tiles
constexpr uint32_t WO_GX = 8;      // WO cores: (x, y), x < 8, y < 8 -> w = 8 y + x
constexpr uint32_t WO_CHUNK = 4;   // k rows per weight push (8 tiles)
constexpr uint32_t NWCH = KWO / WO_CHUNK;  // 8 chunks
constexpr uint32_t BF16_TILE = 2048;
constexpr uint32_t FP32_TILE = 4096;

}  // namespace motif_aout
