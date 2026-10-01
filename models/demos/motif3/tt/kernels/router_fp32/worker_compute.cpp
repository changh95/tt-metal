// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 exact-fp32 router logits (WAVE_A_REVIEW D1(b)), worker compute.
//
// Partial logits of one 128-expert x 128-k block for each tile row of 32 tokens, accumulated on the SFPU in fp32 (no
// FPU sums: the FPU's matmul partial sums are TF32-class, G5). One launch handles n_rows tile rows (common RT arg):
// n_rows = 1 is the decode call ([1, 1, 32, 4096]), n_rows > 1 a prefill chunk ([1, 1, 32 n_rows, 4096]).
//
// DEST (fp32, full sync, 8 tiles; dst_reg index i = DEST rows 4 (i / 2) .. + 3, column parity i % 2, 32 per tile):
//   tiles 0..3  accumulators: slot 4 t + j (token t, vector j); lane 8 r + c = local expert 32 j + 16 (r >> 1)
//               + 2 c + (r & 1) (router_fp32.lane_expert)
//   tiles 4..7  weights of the current chunk (copy_tile of 4 bf16 tiles, exact): slot 4 kc + j = W[k0 + kc, same
//               expert lanes as accumulator j]
// Phase 1 of a row, per chunk (32 k): for every token t: the 4 accumulators live in LREGs; for kc = 0..31:
// x[t, k0 + kc] (bf16 bits read from the x tile in L1 by this math RISC-V) -> SFPLOADI (broadcast to all lanes) ->
// 4 x (SFPLOAD weight, SFPMAD). The product of two bf16 values is exact in fp32, so each MAD rounds once (the add).
// Each chunk is summed from zero and added to the running total in DEST: total = ((c_0 + c_1) + c_2) + c_3.
// After the 4 chunks, SFPTRANSP groups of 4 tokens: register i then holds 4 tokens x 8 experts of face column i >> 1,
// parity i & 1 of output tile j -> stored at its natural place in DEST tile 4 + j; tiles 4..7 are packed (fp32).
// The weights stay in CB_W (L1) for the whole launch; every row copies them into DEST again (16 tiles per row).
//
// Phase 2 of a row (reduce-scatter): after the writers of the expert group have sent their pieces, the reader publishes
// the 32 pieces received for this core's half-face (CB_RECV, fp32, UnpackToDestFp32: exact) -> balanced pairwise fp32
// tree over the k groups (fixed order) -> one 512 B piece of the final logits (CB_FIN).
//
// Row order (n_rows > 1, pipelined): P1(0), P1(1), P2(0), P1(2), P2(1), ..., P1(n-1), P2(n-2), P2(n-1): the
// reduce-scatter of row r is in flight while this core computes row r + 1. The host gives such programs 4 receive
// buffers and 4 semaphores (row r uses buffer / semaphore r % 4): a sender can only reach row r + 4 after this core
// has consumed row r (sender P1(r+4) > its P2(r+2) > this core's send of row r+2 > this core's P1(r+2) > this core's
// P2(r)). n_rows = 1 programs have one receive buffer (P1(0), P2(0)).
// The math RISC-V reads the x tile words itself (SFPLOADI immediates), which the CB protocol does not track: before
// UNPACK pops a row's x tiles (so that the reader may refill them), MATH posts "row done" through the MATH -> UNPACK
// mailbox (paired in program order with the LLK's own unpack-to-dest mailbox messages of phase 2).
//
// CT args: 0 cb_x, 1 cb_w, 2 cb_out, 3 mode (0 logits [replay loop], 1 debug: pack chunk-0 weight tiles, 2 debug:
//          acc = x[t, k0 + j], 3 logits with the SFPI-scheduled loop),
//          4 timing (0/1, timing.h), 5 cb_time, 6 cb_recv, 7 cb_fin, 8 recv_tiles (per row)
// common RT args: 0 n_rows

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "timing.h"

#ifdef TRISC_MATH
#include "ckernel_addrmod.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "lltt.h"

namespace ckernel::sfpu {
namespace motif_router {

using sfpi::dst_reg;
using sfpi::vFloat;

constexpr int W_BASE = 128;  // dst_reg index of weight slot 0 (DEST tile 4)

// acc_j += bf16(bits) * W[kc, j] for the 4 vectors j (one SFPLOADI, 4 SFPLOAD + SFPMAD).
template <int KC>
sfpi_inline void kstep(vFloat& a0, vFloat& a1, vFloat& a2, vFloat& a3, const uint32_t bits) {
    const vFloat xv = vFloat(sfpi::sFloat16b(bits));
    a0 = xv * vFloat(dst_reg[W_BASE + 4 * KC + 0]) + a0;
    a1 = xv * vFloat(dst_reg[W_BASE + 4 * KC + 1]) + a1;
    a2 = xv * vFloat(dst_reg[W_BASE + 4 * KC + 2]) + a2;
    a3 = xv * vFloat(dst_reg[W_BASE + 4 * KC + 3]) + a3;
}

// Word I (0..15) of token t's row in the x tile: I < 8 -> face (k 0..15) word I, else the next face (+512 bytes).
// A word holds kc = 2 I (low half) and 2 I + 1 (high half).
template <int I>
sfpi_inline void kwords(vFloat& a0, vFloat& a1, vFloat& a2, vFloat& a3, const volatile uint32_t* p) {
    constexpr int OFF = (I < 8) ? I : (128 + I - 8);
    const uint32_t w = p[OFF];
    kstep<2 * I>(a0, a1, a2, a3, w & 0xFFFFu);
    kstep<2 * I + 1>(a0, a1, a2, a3, w >> 16);
    if constexpr (I + 1 < 16) {
        kwords<I + 1>(a0, a1, a2, a3, p);
    }
}

// x tile (bf16, faces of 16 x 16): token t's k 0..15 at byte 1024 (t / 16) + 32 (t % 16), k 16..31 at +512.
sfpi_inline const volatile uint32_t* x_row(const uint32_t x_addr, const uint32_t t) {
    return reinterpret_cast<const volatile uint32_t*>(x_addr + 1024u * (t >> 4) + 32u * (t & 15u));
}

// SFPI reference version of the token loop (mode 3; same arithmetic and order as mode 0, compiler-scheduled; the
// RISC-V issues every instruction, ~16 cycles per k step: the JIT build does not fuse .ttinsn).
// Each chunk's 32 products are summed from zero; the chunk sum is then added to the running total in DEST:
// total = ((c_0 + c_1) + c_2) + c_3 (shorter fp32 chains than one 128-term running sum).
template <bool FIRST>
inline void chunk_tokens_sfpi(const uint32_t x_addr) {
#pragma GCC unroll 0
    for (uint32_t t = 0; t < 32; ++t) {
        const volatile uint32_t* p = x_row(x_addr, t);
        vFloat a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        kwords<0>(a0, a1, a2, a3, p);
        if constexpr (!FIRST) {
            a0 = vFloat(dst_reg[4 * t + 0]) + a0;
            a1 = vFloat(dst_reg[4 * t + 1]) + a1;
            a2 = vFloat(dst_reg[4 * t + 2]) + a2;
            a3 = vFloat(dst_reg[4 * t + 3]) + a3;
        }
        dst_reg[4 * t + 0] = a0;
        dst_reg[4 * t + 1] = a1;
        dst_reg[4 * t + 2] = a2;
        dst_reg[4 * t + 3] = a3;
    }
}

// ---- replay version (mode 0) ----------------------------------------------------------------------------------------
// Fixed registers: L0..L3 accumulators of the token, L4 / L7 = x[t, k] for even / odd k (SFPLOADI, bf16 immediate),
// L5 / L6 weight temporaries. The 8 static instructions of a k step (4 x (SFPLOAD weight, SFPMAD)) live in the replay
// buffer (sequence A reads L4, B reads L7); the weight SFPLOADs walk the 128 weight slots through RWC_D with
// ADDR_MOD_6 (+2 rows = one slot per load), so the RISC-V pushes 3 words per k step (SFPLOADI, REPLAY) instead of 9.
// Replay slots 0..15: the SFPU half of the math thread's 32-entry replay buffer (Blackhole LLK: FPU code records at
// cmath_common.h replay_buf_offset = 16, SFPU code at 0). Re-recorded at every chunk, so nothing else may record in
// between (this kernel issues no other replay ops).
constexpr uint32_t REPLAY_A = 0;
constexpr uint32_t REPLAY_B = 8;
constexpr uint32_t REPLAY_LEN = 8;
constexpr uint32_t W_ROW = 256;  // DEST row address of weight slot 0 (dst_reg index 128)

template <uint32_t XR>
inline void kstep_instrs() {
    TTI_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_6, W_ROW);
    TTI_SFPMAD(XR, p_sfpu::LREG5, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_6, W_ROW);
    TTI_SFPMAD(XR, p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LREG1, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_6, W_ROW);
    TTI_SFPMAD(XR, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG2, 0);
    TTI_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_6, W_ROW);
    TTI_SFPMAD(XR, p_sfpu::LREG6, p_sfpu::LREG3, p_sfpu::LREG3, 0);
}

inline void record_ksteps() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);
    load_replay_buf(REPLAY_A, REPLAY_LEN, [] { kstep_instrs<p_sfpu::LREG4>(); });
    load_replay_buf(REPLAY_B, REPLAY_LEN, [] { kstep_instrs<p_sfpu::LREG7>(); });
}

template <bool FIRST>
inline void chunk_tokens(const uint32_t x_addr) {
    record_ksteps();
#pragma GCC unroll 0
    for (uint32_t t = 0; t < 32; ++t) {
        const volatile uint32_t* p = x_row(x_addr, t);
        const uint32_t a = 8 * t;  // DEST row address of accumulator slot 4 t
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        // chunk sum from zero (the running total is added at the end: ((c_0 + c_1) + c_2) + c_3)
        TTI_SFPLOADI(p_sfpu::LREG0, 0, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, 0, 0);
        TTI_SFPLOADI(p_sfpu::LREG2, 0, 0);
        TTI_SFPLOADI(p_sfpu::LREG3, 0, 0);
#pragma GCC unroll 16
        for (uint32_t i = 0; i < 16; ++i) {
            const uint32_t w = p[i < 8 ? i : 120 + i];  // word i of the token's row (i >= 8: next face, +128 words)
            TT_SFPLOADI(p_sfpu::LREG4, 0, w & 0xFFFFu);
            lltt::replay(REPLAY_A, REPLAY_LEN);
            TT_SFPLOADI(p_sfpu::LREG7, 0, w >> 16);
            lltt::replay(REPLAY_B, REPLAY_LEN);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        if constexpr (!FIRST) {  // L_j = total_j * 1.0 + L_j (an exact-product MAD: one fp32 add)
            TT_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_7, a + 0);
            TT_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_7, a + 2);
            TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG0, 0);
            TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG1, 0);
            TT_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_7, a + 4);
            TT_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_7, a + 6);
            TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG2, 0);
            TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG3, 0);
        }
        TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, a + 0);
        TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, a + 2);
        TT_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, a + 4);
        TT_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, a + 6);
    }
}

// mode 2 (debug): acc[t][j] = x[t, k0 + j] in every lane.
inline void debug_x_tokens(const uint32_t x_addr) {
#pragma GCC unroll 0
    for (uint32_t t = 0; t < 32; ++t) {
        const volatile uint32_t* p = x_row(x_addr, t);
        const uint32_t w0 = p[0];
        const uint32_t w1 = p[1];
        dst_reg[4 * t + 0] = vFloat(sfpi::sFloat16b(w0 & 0xFFFFu));
        dst_reg[4 * t + 1] = vFloat(sfpi::sFloat16b(w0 >> 16));
        dst_reg[4 * t + 2] = vFloat(sfpi::sFloat16b(w1 & 0xFFFFu));
        dst_reg[4 * t + 3] = vFloat(sfpi::sFloat16b(w1 >> 16));
    }
}

template <uint32_t MODE>
inline void chunk(const uint32_t x_addr, const uint32_t cc) {
    // The x tile was written into L1 by the reader's NoC transfer: drop any stale L0 data-cache lines first.
    asm volatile("fence" ::: "memory");
    if constexpr (MODE == 2) {
        debug_x_tokens(x_addr);
    } else if constexpr (MODE == 3) {
        if (cc == 0) {
            chunk_tokens_sfpi<true>(x_addr);
        } else {
            chunk_tokens_sfpi<false>(x_addr);
        }
    } else {
        if (cc == 0) {
            chunk_tokens<true>(x_addr);
        } else {
            chunk_tokens<false>(x_addr);
        }
    }
}

// Tokens 4 TQ .. 4 TQ + 3, vector J: SFPTRANSP, then natural positions in DEST tile 4 + J:
// register i -> face 2 (TQ / 4) + (i >> 1), row quad TQ % 4, parity i & 1.
template <int TQ, int J>
sfpi_inline void unpermute_one() {
    vFloat m0 = dst_reg[4 * (4 * TQ + 0) + J];
    vFloat m1 = dst_reg[4 * (4 * TQ + 1) + J];
    vFloat m2 = dst_reg[4 * (4 * TQ + 2) + J];
    vFloat m3 = dst_reg[4 * (4 * TQ + 3) + J];
    sfpi::subvec_transp(m0, m1, m2, m3);
    constexpr int B = 32 * (4 + J) + 16 * (TQ / 4) + 2 * (TQ % 4);
    dst_reg[B + 0] = m0;
    dst_reg[B + 1] = m1;
    dst_reg[B + 8] = m2;
    dst_reg[B + 9] = m3;
}

template <int N>
sfpi_inline void unpermute_from() {
    if constexpr (N < 32) {
        unpermute_one<N / 4, N % 4>();
        unpermute_from<N + 1>();
    }
}

inline void unpermute() { unpermute_from<0>(); }

// Phase 2: DEST tiles 0..3 hold the 32 received pieces (piece h' = 4 SFPU vectors at dst_reg 4 h' .. 4 h' + 3, the
// receive buffer being 32 x 512 B laid out linearly). out[v] = balanced pairwise tree over h' = 0..31 (fp32 adds; same
// tree as the host model router_fp32.emulate_device_fp32) -> DEST tile 4 slots 0..3 (face 0 rows 0..7 = the first
// 512 B of the packed tile).
template <int V, int LO, int N>
sfpi_inline vFloat tree_sum() {
    if constexpr (N == 1) {
        return vFloat(dst_reg[4 * LO + V]);
    } else {
        return tree_sum<V, LO, N / 2>() + tree_sum<V, LO + N / 2, N / 2>();
    }
}

template <int V>
sfpi_inline void sum_piece() {
    dst_reg[128 + V] = tree_sum<V, 0, 32>();
}

inline void sum_pieces() {
    sum_piece<0>();
    sum_piece<1>();
    sum_piece<2>();
    sum_piece<3>();
}

inline void noop_init() {}

}  // namespace motif_router
}  // namespace ckernel::sfpu
#endif  // TRISC_MATH

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t MODE = get_compile_time_arg_val(3);
    constexpr uint32_t TIMING = get_compile_time_arg_val(4);
    constexpr uint32_t cb_time = get_compile_time_arg_val(5);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(6);
    constexpr uint32_t cb_fin = get_compile_time_arg_val(7);
    constexpr uint32_t recv_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t N_CHUNKS = 4;
    constexpr uint32_t W_PER_CHUNK = 4;
    constexpr uint32_t N_VEC = 4;
    static_assert(MODE <= 3, "mode must be 0, 1, 2 or 3");
    const uint32_t n_rows = get_common_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_w, cb_out);
    uint32_t tbase = 0;
    if constexpr (TIMING) {
        tbase = get_tile_address(cb_time, 0);
        UNPACK((motif_router_stamp(tbase, 0)));
        MATH((motif_router_stamp(tbase, 4)));
    }

    bool srca_is_recv = false;  // SrcA format currently configured for CB_RECV (fp32) instead of CB_W (bf16)
    for (uint32_t r = 0; r <= n_rows; ++r) {
        if (r < n_rows) {
            // ---- phase 1 of row r: this core's partial logits (4 fp32 tiles) ----
            if (srca_is_recv) {
                reconfig_data_format_srca(cb_recv, cb_w);
                srca_is_recv = false;
            }
            tile_regs_acquire();
            for (uint32_t cc = 0; cc < N_CHUNKS; ++cc) {
                cb_wait_front(cb_w, W_PER_CHUNK * (cc + 1));  // the weights arrive during row 0 and stay
                if constexpr (TIMING) {
                    if (r == 0 && cc == 0) {
                        UNPACK((motif_router_stamp(tbase, 1)));
                    }
                }
                if (cc > 0) {
                    // The FPU datacopy below overwrites DEST tiles 4..7: wait until the SFPU is done with the previous
                    // chunk.
                    MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)));
                }
                copy_init(cb_w);
                for (uint32_t i = 0; i < W_PER_CHUNK; ++i) {
                    copy_tile(cb_w, W_PER_CHUNK * cc + i, 4 + i);
                }
                if constexpr (MODE == 1) {
                    break;  // DEST tiles 4..7 = chunk-0 weights, packed as they are
                }
                cb_wait_front(cb_x, cc + 1);
                if constexpr (TIMING) {
                    if (r == 0 && cc == 0) {
                        UNPACK((motif_router_stamp(tbase, 2)));
                    } else if (r == 0 && cc == N_CHUNKS - 1) {
                        UNPACK((motif_router_stamp(tbase, 3)));
                    }
                }
                const uint32_t x_addr = get_tile_address(cb_x, cc);
                MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(
                    ckernel::sfpu::motif_router::noop_init)));
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    ckernel::sfpu::motif_router::chunk<MODE>, 0, VectorMode::RC_custom, x_addr, cc)));
                if constexpr (MODE == 2) {
                    break;
                }
            }
            if constexpr (MODE != 1) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    ckernel::sfpu::motif_router::unpermute, 0, VectorMode::RC_custom)));
            }
            if constexpr (TIMING) {
                if (r == 0) {
                    MATH((motif_router_stamp(tbase, 5)));
                }
            }
            // Release the row's x tiles: the math RISC-V has read every x word it needs (they are SFPLOADI immediates
            // already in the instruction stream). UNPACK pops only after MATH says so, then the reader may refill them.
            MATH((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, r)));
            cb_wait_front(cb_x, N_CHUNKS);  // debug modes 1 / 2 have not waited for every x tile
            UNPACK(((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId)));
            cb_pop_front(cb_x, N_CHUNKS);
            tile_regs_commit();

            cb_reserve_back(cb_out, N_VEC);
            tile_regs_wait();
            if constexpr (TIMING) {
                if (r == 0) {
                    PACK((motif_router_stamp(tbase, 6)));
                }
            }
            for (uint32_t j = 0; j < N_VEC; ++j) {
                pack_tile(4 + j, cb_out);
            }
            tile_regs_release();
            if constexpr (TIMING) {
                if (r == 0) {
                    PACK((motif_router_stamp(tbase, 7)));
                }
            }
            cb_push_back(cb_out, N_VEC);
        }

        if (r >= 1) {
            // ---- phase 2 of row r - 1: sum the 32 pieces of this core's half-face (fp32, k-group order) ----
            if (!srca_is_recv) {
                reconfig_data_format_srca(cb_w, cb_recv);
                srca_is_recv = true;
            }
            cb_wait_front(cb_recv, recv_tiles);
            tile_regs_acquire();
            copy_init(cb_recv);
            for (uint32_t i = 0; i < recv_tiles; ++i) {
                copy_tile(cb_recv, i, i);
            }
            MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::motif_router::noop_init)));
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::motif_router::sum_pieces, 0, VectorMode::RC_custom)));
            tile_regs_commit();
            cb_reserve_back(cb_fin, 1);
            tile_regs_wait();
            pack_tile(4, cb_fin);
            tile_regs_release();
            cb_push_back(cb_fin, 1);
            cb_pop_front(cb_recv, recv_tiles);
        }
    }
    cb_wait_front(cb_w, W_PER_CHUNK * N_CHUNKS);
    cb_pop_front(cb_w, W_PER_CHUNK * N_CHUNKS);
}
