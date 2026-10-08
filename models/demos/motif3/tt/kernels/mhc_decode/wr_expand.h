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
#ifndef MHC_EXPAND_SKIP
#define MHC_EXPAND_SKIP 0
#endif

namespace {
constexpr uint32_t FP32_TILE_WORDS = 1024;
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
