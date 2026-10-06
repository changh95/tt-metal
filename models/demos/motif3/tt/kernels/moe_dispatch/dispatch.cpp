// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 compacted prefill MoE: on-device local dispatch (B2b, docs/OPTIMIZATION_PLAN.md §3.3 B2), one data-movement
// kernel (RISCV_0) on E + 1 cores. It builds on device, from the router's routes, the row lists that B2a's host pass
// (tt/moe.py compact_upload_fast) uploads, word for word:
//
//   idx   [1, 1, M, K] uint32 TILE          top-K global expert ids per token (identical on every chip)
//   wsrc  WSRC 0: w_loc [1, E, M, 1] fp32 TILE (MotifMoE.local_weights: the dense path's routing weights)
//         WSRC 1: w [1, 1, M, K] fp32 TILE (the router's weights, same order as idx)
//   slot  [1, 1, 1, NE] uint32 ROW_MAJOR      global id -> chip-expert slot p E + e (replicated)
//   cmeta [1, 1, 1, 64] uint32 ROW_MAJOR      per chip: [0] p, [1 .. E] the chip's global ids, [16 + 4 e + i] the bf16
//                                             bit pattern of expert e's PolyNorm constant i (c0, c1, c2, b)
// ->
//   rows  [1, 1, 2, CAP MB] uint32 ROW_MAJOR  page 0: the token of every row (pad 0), page 1: the combine key (the
//                                             token; pad rows M)
//   wcol  [1, CAP, MB, 1] fp32 TILE           the routing weight of every row (column 0; pad rows and columns 1.. 0)
//   sp    [1, CAP, 1, 32] bf16 ROW_MAJOR      per block: one-hot of its local expert (bf16 1.0), columns 12.. 0
//   blk   [1, CAP, 1, 32] bf16 TILE           per block, row 0: [one-hot (E) | 0 | c0 c1 c2 b | 0] (B2a's block word)
//   need  [1, 1, 1, 8] uint32 ROW_MAJOR       [0] blocks the busiest chip needs, [1] NB (the ladder entry >= need, 0
//                                             beyond the cap), [2] blocks this chip uses
//
// Rows are sorted by (local expert, token): expert e's rows start at block off_e (sum of the lower experts' blocks,
// blocks of MB rows), its tokens ascending; blocks [used, NB) are pad blocks of expert E - 1 (B2a's layout). Only
// blocks [0, NB) are written (an expert core writes when its chip's blocks fit the cap); the host slices that prefix
// after reading `need` and ignores everything when NB = 0.
//
// Core q (row-major over GX columns): q < E handles local expert q: one pass over the routes counts this chip's 12
// experts (its offset) and lists its own tokens; then it fetches their weights and writes its rows and blocks. q == E
// counts every expert (every chip's block count: need, NB), writes the pad blocks and `need`. The routes are read once
// per core into L1 (all reads issued, one barrier). Pure integer work and bit copies: the outputs equal the host lists
// exactly.
//
// CT: 0 M, 1 K, 2 E, 3 P, 4 NE, 5 MB, 6 CAP, 7 NLAD, 8 .. 23 ladder (16 slots), 24 GX, 25 WSRC, 26 CONTIG (every
//     chip's ids are base + e: local test without the slot table), 27 cb_idx, 28 cb_w, 29 cb_misc, 30 cb_rows,
//     31 cb_tile, 32.. TensorAccessorArgs: idx, wsrc, slot, cmeta, rows, wcol, sp, blk, need
// common RT: 0 idx, 1 wsrc, 2 slot, 3 cmeta, 4 rows, 5 wcol, 6 sp, 7 blk, 8 need (buffer addresses)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t ladder_at(uint32_t i) {
    return i == 0    ? get_compile_time_arg_val(8)
           : i == 1  ? get_compile_time_arg_val(9)
           : i == 2  ? get_compile_time_arg_val(10)
           : i == 3  ? get_compile_time_arg_val(11)
           : i == 4  ? get_compile_time_arg_val(12)
           : i == 5  ? get_compile_time_arg_val(13)
           : i == 6  ? get_compile_time_arg_val(14)
           : i == 7  ? get_compile_time_arg_val(15)
           : i == 8  ? get_compile_time_arg_val(16)
           : i == 9  ? get_compile_time_arg_val(17)
           : i == 10 ? get_compile_time_arg_val(18)
           : i == 11 ? get_compile_time_arg_val(19)
           : i == 12 ? get_compile_time_arg_val(20)
           : i == 13 ? get_compile_time_arg_val(21)
           : i == 14 ? get_compile_time_arg_val(22)
                     : get_compile_time_arg_val(23);
}

inline uint32_t align64(uint32_t a) { return (a + 63) & ~63u; }  // CB scratch is allocated with 64 B of slack

inline void fence_l1() {
    invalidate_l1_cache();
    asm volatile("fence" ::: "memory");
}

// element (r, c < 16) of a 32 x 32 tile of 4-byte words staged as [face 0 | face 2] (2 KB): rows 0..15, then 16..31
inline uint32_t face_word(const uint32_t* st, uint32_t r, uint32_t c) { return st[(r >> 4) * 256 + (r & 15) * 16 + c]; }
}  // namespace

void kernel_main() {
    constexpr uint32_t M = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t P = get_compile_time_arg_val(3);
    constexpr uint32_t NE = get_compile_time_arg_val(4);
    constexpr uint32_t MB = get_compile_time_arg_val(5);
    constexpr uint32_t CAP = get_compile_time_arg_val(6);
    constexpr uint32_t NLAD = get_compile_time_arg_val(7);
    constexpr uint32_t GX = get_compile_time_arg_val(24);
    constexpr uint32_t WSRC = get_compile_time_arg_val(25);
    constexpr uint32_t CONTIG = get_compile_time_arg_val(26);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(27);
    constexpr uint32_t cb_w = get_compile_time_arg_val(28);
    constexpr uint32_t cb_misc = get_compile_time_arg_val(29);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(30);
    constexpr uint32_t cb_tile = get_compile_time_arg_val(31);
    constexpr auto idx_args = TensorAccessorArgs<32>();
    constexpr auto w_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto slot_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto cm_args = TensorAccessorArgs<slot_args.next_compile_time_args_offset()>();
    constexpr auto rows_args = TensorAccessorArgs<cm_args.next_compile_time_args_offset()>();
    constexpr auto wcol_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();
    constexpr auto sp_args = TensorAccessorArgs<wcol_args.next_compile_time_args_offset()>();
    constexpr auto blk_args = TensorAccessorArgs<sp_args.next_compile_time_args_offset()>();
    constexpr auto need_args = TensorAccessorArgs<blk_args.next_compile_time_args_offset()>();
    static_assert(M % 32 == 0 && MB % 32 == 0 && K <= 16 && E <= 12 && P * E <= 1024 && NE <= 1024, "dispatch shape");
    static_assert(NLAD >= 1 && NLAD <= 16, "ladder size");
    static_assert(WSRC == 0 || WSRC == 1, "weight source");

    constexpr uint32_t MT = M / 32;          // token tile rows
    constexpr uint32_t ROWS_CAP = CAP * MB;  // rows of the output buffers
    constexpr uint32_t TPB = MB / 32;        // wcol tiles per block

    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();

    const auto idx_s = TensorAccessor(idx_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto w_s = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1), 4096);
    const auto slot_s = TensorAccessor(slot_args, get_common_arg_val<uint32_t>(2), NE * 4);
    const auto cm_s = TensorAccessor(cm_args, get_common_arg_val<uint32_t>(3), 256);
    const auto rows_s = TensorAccessor(rows_args, get_common_arg_val<uint32_t>(4), ROWS_CAP * 4);
    const auto wcol_s = TensorAccessor(wcol_args, get_common_arg_val<uint32_t>(5), 4096);
    const auto sp_s = TensorAccessor(sp_args, get_common_arg_val<uint32_t>(6), 64);
    const auto blk_s = TensorAccessor(blk_args, get_common_arg_val<uint32_t>(7), 2048);
    const auto need_s = TensorAccessor(need_args, get_common_arg_val<uint32_t>(8), 32);

    // ---- L1 scratch (CBs used as plain memory) ----
    cb_reserve_back(cb_idx, 1);
    cb_reserve_back(cb_w, 1);
    cb_reserve_back(cb_misc, 1);
    cb_reserve_back(cb_rows, 1);
    cb_reserve_back(cb_tile, 1);
    const uint32_t idx_l1 = align64(get_write_ptr(cb_idx));  // MT x [face 0 | face 2] of the idx tiles (2 KB each)
    const uint32_t w_l1 = align64(get_write_ptr(cb_w));      // MT x 2 KB: this expert's weight faces (by tile row)
    const uint32_t misc = align64(get_write_ptr(cb_misc));
    const uint32_t slot_l1 = misc;                         // NE words
    const uint32_t cnt_l1 = misc + 4096;                   // NE words (q == E) / E words (q < E)
    const uint32_t cm_l1 = misc + 8192;                    // 64 words
    const uint32_t sp_l1 = cm_l1 + 256;                    // 64 B
    const uint32_t need_l1 = sp_l1 + 64;                   // 32 B
    const uint32_t chip_l1 = need_l1 + 64;                 // P words (q == E)
    const uint32_t tok_l1 = misc + 16384;                  // M words: this expert's tokens (q < E)
    const uint32_t rows_l1 = align64(get_write_ptr(cb_rows));  // tix [ROWS_CAP], keys [ROWS_CAP]
    const uint32_t tile_l1 = align64(get_write_ptr(cb_tile));  // wcol tile 4 KB | blk tile 2 KB
    const uint32_t wt_l1 = tile_l1;
    const uint32_t blk_l1 = tile_l1 + 4096;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);

    noc_async_read(slot_s.get_noc_addr(0, 0), slot_l1, NE * 4);
    noc_async_read(cm_s.get_noc_addr(0, 0), cm_l1, 256);
    for (uint32_t tr = 0; tr < MT; ++tr) {
        noc_async_read(idx_s.get_noc_addr(tr, 0), idx_l1 + tr * 2048, 1024);            // face 0: rows 0..15
        noc_async_read(idx_s.get_noc_addr(tr, 2048), idx_l1 + tr * 2048 + 1024, 1024);  // face 2: rows 16..31
    }
    noc_async_read_barrier();
    fence_l1();
    const uint32_t* slot = reinterpret_cast<const uint32_t*>(slot_l1);
    const uint32_t* cm = reinterpret_cast<const uint32_t*>(cm_l1);
    const uint32_t* ids = reinterpret_cast<const uint32_t*>(idx_l1);
    uint32_t* cnt = reinterpret_cast<uint32_t*>(cnt_l1);
    const uint32_t my_p = cm[0];
    const uint32_t base = cm[1];

    if (q < E) {
        // ---- expert core: count this chip's experts, list expert q's tokens (ascending) ----
        const uint32_t e = q;
        for (uint32_t d = 0; d < E; ++d) {
            cnt[d] = 0;
        }
        uint32_t* tok = reinterpret_cast<uint32_t*>(tok_l1);
        uint32_t n = 0;
        for (uint32_t tr = 0; tr < MT; ++tr) {
            const uint32_t* st = ids + tr * 512;
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t* row = st + (r >> 4) * 256 + (r & 15) * 16;
                for (uint32_t k = 0; k < K; ++k) {
                    uint32_t d;
                    if constexpr (CONTIG) {
                        d = row[k] - base;  // unsigned: other chips' ids land >= E
                    } else {
                        d = slot[row[k]] - my_p * E;
                    }
                    if (d < E) {
                        cnt[d] += 1;
                        if (d == e) {
                            tok[n++] = (tr << 9) | (r << 4) | k;  // token tile row, row, rank of the hit
                        }
                    }
                }
            }
        }
        uint32_t off = 0;
        uint32_t used = 0;
        for (uint32_t d = 0; d < E; ++d) {
            const uint32_t b = (cnt[d] + MB - 1) / MB;
            off += d < e ? b : 0;
            used += b;
        }
        const uint32_t nblk = (n + MB - 1) / MB;
        if (nblk != 0 && used <= CAP) {
            // weights: fetch the faces of every tile row holding one of the tokens (one barrier)
            uint32_t last = 0xFFFFFFFFu;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t tr = tok[i] >> 9;
                if (tr != last) {
                    const uint32_t page = WSRC == 0 ? e * MT + tr : tr;
                    noc_async_read(w_s.get_noc_addr(page, 0), w_l1 + tr * 2048, 1024);
                    noc_async_read(w_s.get_noc_addr(page, 2048), w_l1 + tr * 2048 + 1024, 1024);
                    last = tr;
                }
            }
            noc_async_read_barrier();
            fence_l1();
            const uint32_t* wf = reinterpret_cast<const uint32_t*>(w_l1);
            const uint32_t nrows = nblk * MB;
            const uint32_t row0 = off * MB;
            uint32_t* tix = reinterpret_cast<uint32_t*>(rows_l1);
            uint32_t* key = reinterpret_cast<uint32_t*>(rows_l1 + ROWS_CAP * 4);
            uint32_t* wt = reinterpret_cast<uint32_t*>(wt_l1);
            for (uint32_t tj = 0; tj < nblk * TPB; ++tj) {
                // one wcol tile: zero it, set column 0 of its rows, write it
                for (uint32_t i = 0; i < 8; ++i) {
                    noc_async_read(zeros, wt_l1 + i * 512, 512);
                }
                noc_async_read_barrier();
                fence_l1();
                const uint32_t j0 = tj * 32;
                const uint32_t j1 = j0 + 32 < n ? j0 + 32 : n;
                for (uint32_t j = j0; j < j1; ++j) {
                    const uint32_t v = tok[j];
                    const uint32_t tr = v >> 9;
                    const uint32_t r = (v >> 4) & 31;
                    const uint32_t t = tr * 32 + r;
                    tix[j] = t;
                    key[j] = t;
                    const uint32_t wbits = face_word(wf + tr * 512, r, WSRC == 0 ? 0 : (v & 15));
                    const uint32_t jr = j & 31;
                    wt[((jr >> 4) * 2) * 256 + (jr & 15) * 16] = wbits;
                }
                for (uint32_t j = (j1 > j0 ? j1 : j0); j < j0 + 32; ++j) {
                    tix[j] = 0;
                    key[j] = M;
                }
                asm volatile("" ::: "memory");
                noc_async_write(wt_l1, wcol_s.get_noc_addr(row0 / 32 + tj, 0), 4096);
                noc_async_write_barrier();
            }
            asm volatile("" ::: "memory");
            noc_async_write(rows_l1, rows_s.get_noc_addr(0, row0 * 4), nrows * 4);
            noc_async_write(rows_l1 + ROWS_CAP * 4, rows_s.get_noc_addr(1, row0 * 4), nrows * 4);
            // block words: sparsity (ROW_MAJOR page) and the TILE block row (one-hot + PolyNorm constants)
            uint16_t* sp = reinterpret_cast<uint16_t*>(sp_l1);
            for (uint32_t i = 0; i < 32; ++i) {
                sp[i] = i == e ? 0x3F80 : 0;
            }
            for (uint32_t i = 0; i < 4; ++i) {
                noc_async_read(zeros, blk_l1 + i * 512, 512);
            }
            noc_async_read_barrier();
            fence_l1();
            uint16_t* bk = reinterpret_cast<uint16_t*>(blk_l1);
            bk[e] = 0x3F80;  // face 0, row 0, column e
            for (uint32_t i = 0; i < 4; ++i) {
                bk[256 + i] = static_cast<uint16_t>(cm[16 + 4 * e + i] & 0xFFFF);  // face 1, row 0, column 16 + i
            }
            asm volatile("" ::: "memory");
            for (uint32_t b = off; b < off + nblk; ++b) {
                noc_async_write(sp_l1, sp_s.get_noc_addr(b, 0), 64);
                noc_async_write(blk_l1, blk_s.get_noc_addr(b, 0), 2048);
            }
            noc_async_write_barrier();
        }
    } else if (q == E) {
        // ---- every expert's count -> every chip's blocks -> need, NB; pad blocks [used, NB); need ----
        for (uint32_t g = 0; g < NE; ++g) {
            cnt[g] = 0;
        }
        for (uint32_t tr = 0; tr < MT; ++tr) {
            const uint32_t* st = ids + tr * 512;
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t* row = st + (r >> 4) * 256 + (r & 15) * 16;
                for (uint32_t k = 0; k < K; ++k) {
                    const uint32_t g = row[k];
                    cnt[g < NE ? g : 0] += 1;
                }
            }
        }
        uint32_t* chip = reinterpret_cast<uint32_t*>(chip_l1);
        for (uint32_t p = 0; p < P; ++p) {
            chip[p] = 0;
        }
        for (uint32_t g = 0; g < NE; ++g) {
            chip[slot[g] / E] += (cnt[g] + MB - 1) / MB;
        }
        uint32_t need = 0;
        for (uint32_t p = 0; p < P; ++p) {
            need = chip[p] > need ? chip[p] : need;
        }
        const uint32_t used = chip[my_p];
        uint32_t nb = 0;
        for (uint32_t i = 0; i < NLAD; ++i) {
            const uint32_t l = ladder_at(i);
            if (l >= need) {
                nb = l;
                break;
            }
        }
        if (nb != 0 && used < nb) {
            const uint32_t row0 = used * MB;
            const uint32_t nrows = (nb - used) * MB;
            uint32_t* tix = reinterpret_cast<uint32_t*>(rows_l1);
            uint32_t* key = reinterpret_cast<uint32_t*>(rows_l1 + ROWS_CAP * 4);
            for (uint32_t j = 0; j < nrows; ++j) {
                tix[j] = 0;
                key[j] = M;
            }
            uint16_t* sp = reinterpret_cast<uint16_t*>(sp_l1);
            for (uint32_t i = 0; i < 32; ++i) {
                sp[i] = i == E - 1 ? 0x3F80 : 0;
            }
            for (uint32_t i = 0; i < 8; ++i) {
                noc_async_read(zeros, wt_l1 + i * 512, 512);
            }
            for (uint32_t i = 0; i < 4; ++i) {
                noc_async_read(zeros, blk_l1 + i * 512, 512);
            }
            noc_async_read_barrier();
            fence_l1();
            uint16_t* bk = reinterpret_cast<uint16_t*>(blk_l1);
            bk[E - 1] = 0x3F80;
            for (uint32_t i = 0; i < 4; ++i) {
                bk[256 + i] = static_cast<uint16_t>(cm[16 + 4 * (E - 1) + i] & 0xFFFF);
            }
            asm volatile("" ::: "memory");
            noc_async_write(rows_l1, rows_s.get_noc_addr(0, row0 * 4), nrows * 4);
            noc_async_write(rows_l1 + ROWS_CAP * 4, rows_s.get_noc_addr(1, row0 * 4), nrows * 4);
            for (uint32_t tj = row0 / 32; tj < nb * TPB; ++tj) {
                noc_async_write(wt_l1, wcol_s.get_noc_addr(tj, 0), 4096);
            }
            for (uint32_t b = used; b < nb; ++b) {
                noc_async_write(sp_l1, sp_s.get_noc_addr(b, 0), 64);
                noc_async_write(blk_l1, blk_s.get_noc_addr(b, 0), 2048);
            }
        }
        uint32_t* nd = reinterpret_cast<uint32_t*>(need_l1);
        nd[0] = need;
        nd[1] = nb;
        nd[2] = used;
        for (uint32_t i = 3; i < 8; ++i) {
            nd[i] = 0;
        }
        asm volatile("" ::: "memory");
        noc_async_write(need_l1, need_s.get_noc_addr(0, 0), 32);
        noc_async_write_barrier();
    }
}
