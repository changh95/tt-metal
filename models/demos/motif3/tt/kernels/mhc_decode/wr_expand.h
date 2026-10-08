// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 mHC decode stream mixing (D3): the packed-tile -> weight-CB expansion shared by wr_reader.cpp and
// wr_writer.cpp (MHC_EXPAND_SPLIT: the writer RISC expands the second half of the set).

#pragma once

#include <cstdint>

#ifndef MHC_EXPAND_SPLIT
#define MHC_EXPAND_SPLIT 0
#endif
#ifndef MHC_EXPAND_ZERO_HI
#define MHC_EXPAND_ZERO_HI 0
#endif
#ifndef MHC_P_READ_MIN
#define MHC_P_READ_MIN 0
#endif
#ifndef MHC_P_READ_NONE
#define MHC_P_READ_NONE 0
#endif
#ifndef MHC_WR_ITEMS
#define MHC_WR_ITEMS 0
#endif
#ifndef MHC_EXPAND_SKIP
#define MHC_EXPAND_SKIP 0
#endif

namespace {
constexpr uint32_t FP32_TILE_WORDS = 1024;

// Work split (MHC_WR_ITEMS): items are positions (0: every core does all NUM_R output rows of its positions, the
// release split) or (position, output row) pairs, item = p * NUM_R + r (1: the MACs spread evenly over the cores; a
// core covers at most a few positions, each with a contiguous row range [r_lo, r_hi)).
struct WrSplit {
    uint32_t start, n;  // items [start, start + n)
};
inline WrSplit wr_split(uint32_t total, uint32_t num_cores, uint32_t core_i) {
    const uint32_t q = total / num_cores, rem = total % num_cores;
    return {core_i * q + (core_i < rem ? core_i : rem), q + (core_i < rem ? 1u : 0u)};
}
template <uint32_t ITEMS, uint32_t NUM_R>
constexpr uint32_t items_per_position() {
    return ITEMS ? NUM_R : 1;
}
template <uint32_t ITEMS, uint32_t NUM_R>
inline void row_range(const WrSplit& s, uint32_t p, uint32_t& r_lo, uint32_t& r_hi) {
    if constexpr (ITEMS) {
        const uint32_t a = p * NUM_R, b = a + NUM_R;
        r_lo = (s.start > a ? s.start : a) - a;
        r_hi = ((s.start + s.n) < b ? (s.start + s.n) : b) - a;
    } else {
        r_lo = 0;
        r_hi = NUM_R;
    }
}
template <uint32_t HAS_OUT, uint32_t NUM_X>
constexpr uint32_t packed_row(uint32_t r, uint32_t c) {
    if constexpr (HAS_OUT == 0) {
        return c;
    } else {
        return c < NUM_X ? 8 + 4 * r + c : 4 + r;
    }
}

// Expand weight j = r * NUM_C + c of the set from the packed tile p into tile j of the weight CB at w.
template <uint32_t HAS_OUT, uint32_t NUM_X, uint32_t NUM_C>
inline __attribute__((always_inline)) void expand_weight(uint32_t* __restrict w, const uint32_t* __restrict p,
                                                         uint32_t j) {
    const uint32_t r = j / NUM_C, c = j % NUM_C;
    const uint32_t k = packed_row<HAS_OUT, NUM_X>(r, c);
    // row k of P: tokens 0-15 in face 2 (k / 16), tokens 16-31 in the next face
    const uint32_t* __restrict src = p + (k >> 4) * 512 + (k & 15) * 16;
    uint32_t* __restrict dst = w + j * FP32_TILE_WORDS;
    uint32_t v[16];
#pragma GCC unroll 16
    for (uint32_t t = 0; t < 16; ++t) {
        v[t] = src[t];
    }
#pragma GCC unroll 16
    for (uint32_t t = 0; t < 16; ++t) {
        dst[16 * t] = v[t];
    }
#if MHC_EXPAND_ZERO_HI
#pragma GCC unroll 16
    for (uint32_t t = 0; t < 16; ++t) {
        dst[512 + 16 * t] = 0;
    }
#else
#pragma GCC unroll 16
    for (uint32_t t = 0; t < 16; ++t) {
        v[t] = src[256 + t];
    }
#pragma GCC unroll 16
    for (uint32_t t = 0; t < 16; ++t) {
        dst[512 + 16 * t] = v[t];
    }
#endif
}
}  // namespace
