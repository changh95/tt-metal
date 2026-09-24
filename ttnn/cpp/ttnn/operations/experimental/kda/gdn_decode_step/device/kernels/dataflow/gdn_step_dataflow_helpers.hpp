// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Dataflow helpers shared by the gdn_decode_step reader/writer kernels.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace gdn_step_df {

template <typename Accessor>
inline void read_tiles_at(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count, uint32_t dst_tile) {
    const uint32_t entry = dfb.get_entry_size();
    for (uint32_t t = 0; t < count; ++t) {
        noc.async_read(acc, dfb, entry, {.page_id = first_page + t}, {.offset_bytes = (dst_tile + t) * entry});
    }
}

template <typename Accessor>
inline void read_tiles(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count) {
    dfb.reserve_back(count);
    read_tiles_at(acc, dfb, noc, first_page, count, 0);
    noc.async_read_barrier();
    dfb.push_back(count);
}

// Read the head's [q | k | v] tiles of a row tensor laid out [q(Nk*Dk) | k(Nk*Dk) | v(Nv*Dv) | ...] into one DFB.
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void read_head_row(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h) {
    dfb.reserve_back(2 * Kt + Vt);
    read_tiles_at(acc, dfb, noc, hk * Kt, Kt, 0);
    read_tiles_at(acc, dfb, noc, Nk * Kt + hk * Kt, Kt, Kt);
    read_tiles_at(acc, dfb, noc, 2 * Nk * Kt + h * Vt, Vt, 2 * Kt);
    noc.async_read_barrier();
    dfb.push_back(2 * Kt + Vt);
}

// Write the head's [q | k | v] tiles held in `dfb` back to a row tensor (q/k only when `write_qk`).
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void write_head_row(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h, bool write_qk) {
    constexpr uint32_t Ct = 2 * Kt + Vt;
    dfb.wait_front(Ct);
    const uint32_t entry = dfb.get_entry_size();
    if (write_qk) {
        for (uint32_t t = 0; t < Kt; ++t) {
            noc.async_write(dfb, acc, entry, {.offset_bytes = t * entry}, {.page_id = hk * Kt + t});
            noc.async_write(dfb, acc, entry, {.offset_bytes = (Kt + t) * entry}, {.page_id = Nk * Kt + hk * Kt + t});
        }
    }
    for (uint32_t t = 0; t < Vt; ++t) {
        noc.async_write(dfb, acc, entry, {.offset_bytes = (2 * Kt + t) * entry}, {.page_id = 2 * Nk * Kt + h * Vt + t});
    }
    noc.async_write_barrier();
    dfb.pop_front(Ct);
}

FORCE_INLINE uint32_t tile_elem_index(uint32_t row, uint32_t col) {
    // 32x32 tile stored as four 16x16 faces: f0 (r<16,c<16), f1 (r<16,c>=16), f2, f3
    return ((row < 16 ? 0u : 2u) + (col < 16 ? 0u : 1u)) * 256u + (row & 15u) * 16u + (col & 15u);
}

// Broadcast one fp32 value over a whole fp32 tile in `dfb` (slot must be reserved; lock covers 1 entry).
inline void fill_scalar_tile(DataflowBuffer& dfb, uint32_t value) {
    auto lock = dfb.scoped_write_lock(1);
    auto p32 = lock.template get_ptr<volatile uint32_t>();
    for (uint32_t i = 0; i < 1024; ++i) {
        p32[i] = value;
    }
}

// Same fill, 8x unrolled (the loop overhead, not the store, bounds a volatile fill: ~5 us less per user at B=1).
// Used only on single-user cores: on the 96-core width-32 launch the denser L1 store burst measured slightly slower
// in the served step (+0.1..0.2 ms) although faster in isolation.
inline void fill_scalar_tile_fast(DataflowBuffer& dfb, uint32_t value) {
    auto lock = dfb.scoped_write_lock(1);
    auto p32 = lock.template get_ptr<volatile uint32_t>();
    for (uint32_t i = 0; i < 1024; i += 8) {
        p32[i] = value;
        p32[i + 1] = value;
        p32[i + 2] = value;
        p32[i + 3] = value;
        p32[i + 4] = value;
        p32[i + 5] = value;
        p32[i + 6] = value;
        p32[i + 7] = value;
    }
}

// Element (row, col) of the tile already read into the reserved slot of `dfb`, as fp32 bits (bf16 source widened).
template <bool src_fp32>
FORCE_INLINE uint32_t tile_scalar_bits(DataflowBuffer& dfb, uint32_t row, uint32_t col) {
    auto lock = dfb.scoped_write_lock(1);
    const uint32_t idx = tile_elem_index(row, col);
    if constexpr (src_fp32) {
        auto p = lock.template get_ptr<volatile uint32_t>();
        return p[idx];
    } else {
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        return static_cast<uint32_t>(p16[idx]) << 16;
    }
}

// Read tile `page` of a tensor and return element (row 0, col) as fp32 bits (bf16 source is widened).
template <bool src_fp32, typename Accessor>
FORCE_INLINE uint32_t
load_tile_scalar(const Accessor& acc, DataflowBuffer& staging, Noc& noc, uint32_t page, uint32_t col) {
    constexpr uint32_t src_bytes = src_fp32 ? 4096 : 2048;
    noc.async_read(acc, staging, src_bytes, {.page_id = page}, {.offset_bytes = 0});
    noc.async_read_barrier();
    auto lock = staging.scoped_write_lock(1);
    const uint32_t idx = tile_elem_index(0, col);
    if constexpr (src_fp32) {
        auto p = lock.template get_ptr<volatile uint32_t>();
        return p[idx];
    } else {
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        return static_cast<uint32_t>(p16[idx]) << 16;
    }
}

// Read tile 0 of a [.., H]-wide tensor and broadcast element (row 0, col) over an fp32 tile pushed into `dfb`.
template <bool src_fp32, typename Accessor>
inline void load_head_scalar(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t col) {
    dfb.reserve_back(1);
    const uint32_t value = load_tile_scalar<src_fp32>(acc, dfb, noc, 0, col);
    fill_scalar_tile(dfb, value);
    dfb.push_back(1);
}

// bf16 column mask: 1.0 in row 0 (column 0), 0 elsewhere.
inline void build_row0_mask(DataflowBuffer& dfb, Noc& noc) {
    dfb.reserve_back(1);
    noc.async_write_zeros(dfb, dfb.get_entry_size());
    noc.write_zeros_l1_barrier();
    {
        auto lock = dfb.scoped_write_lock(1);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        p16[0] = 0x3F80;
    }
    dfb.push_back(1);
}

// ---- row-0-only transfers: a 32x32 tile stores row 0 as two 32-byte face rows at byte offsets 0 and entry/4.
// Only the token row matters for the decode inputs, so moving 64 B instead of a whole tile cuts the per-core
// traffic ~30x. Tiles are zero-filled first so the untouched rows are finite (they are masked, not ignored).
template <typename Accessor>
inline void read_row0_at(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t dst_tile) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t seg = entry / 64;  // bytes of one face row: 32 for bf16, 64 for fp32
    const uint32_t base = dst_tile * entry;
    noc.async_read(acc, dfb, seg, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = base});
    noc.async_read(acc, dfb, seg, {.page_id = page, .offset_bytes = entry / 4}, {.offset_bytes = base + entry / 4});
}

template <typename Accessor>
inline void write_row0_at(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t src_tile) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t seg = entry / 64;
    const uint32_t base = src_tile * entry;
    noc.async_write(dfb, acc, seg, {.offset_bytes = base}, {.page_id = page, .offset_bytes = 0});
    noc.async_write(dfb, acc, seg, {.offset_bytes = base + entry / 4}, {.page_id = page, .offset_bytes = entry / 4});
}

inline void zero_reserved(DataflowBuffer& dfb, Noc& noc, uint32_t tiles) {
    noc.async_write_zeros(dfb, tiles * dfb.get_entry_size());
    noc.write_zeros_l1_barrier();
}

// Row 0 of `count` consecutive tiles (tiles zero-filled first when `zero`).
template <typename Accessor>
inline void read_tiles_r0(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count, bool zero = true) {
    dfb.reserve_back(count);
    if (zero) {
        zero_reserved(dfb, noc, count);
    }
    for (uint32_t t = 0; t < count; ++t) {
        read_row0_at(acc, dfb, noc, first_page + t, t);
    }
    noc.async_read_barrier();
    dfb.push_back(count);
}

// Row 0 of the head's [q | k | v] tiles of a [q | k | v | ...] row tensor.
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void read_head_row_r0(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h, bool zero = true) {
    constexpr uint32_t Ct = 2 * Kt + Vt;
    dfb.reserve_back(Ct);
    if (zero) {
        zero_reserved(dfb, noc, Ct);
    }
    for (uint32_t t = 0; t < Kt; ++t) {
        read_row0_at(acc, dfb, noc, hk * Kt + t, t);
        read_row0_at(acc, dfb, noc, Nk * Kt + hk * Kt + t, Kt + t);
    }
    for (uint32_t t = 0; t < Vt; ++t) {
        read_row0_at(acc, dfb, noc, 2 * Nk * Kt + h * Vt + t, 2 * Kt + t);
    }
    noc.async_read_barrier();
    dfb.push_back(Ct);
}

// Row 0 of the head's [q | k | v] tiles held in `dfb` back to a row tensor (q/k only when `write_qk`).
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void write_head_row_r0(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h, bool write_qk) {
    constexpr uint32_t Ct = 2 * Kt + Vt;
    dfb.wait_front(Ct);
    if (write_qk) {
        for (uint32_t t = 0; t < Kt; ++t) {
            write_row0_at(acc, dfb, noc, hk * Kt + t, t);
            write_row0_at(acc, dfb, noc, Nk * Kt + hk * Kt + t, Kt + t);
        }
    }
    for (uint32_t t = 0; t < Vt; ++t) {
        write_row0_at(acc, dfb, noc, 2 * Nk * Kt + h * Vt + t, 2 * Kt + t);
    }
    noc.async_write_barrier();
    dfb.pop_front(Ct);
}

// Copy row 0 of source tile `page` into row `dst_row` of tile `dst_tile` in `dfb` (two face-row segments).
template <typename Accessor>
inline void pack_row_from(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t dst_tile, uint32_t dst_row) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t seg = entry / 64;    // one face row: 32 B for bf16
    const uint32_t esz = entry / 1024;  // bytes per element
    const uint32_t base = dst_tile * entry;
    noc.async_read(
        acc,
        dfb,
        seg,
        {.page_id = page, .offset_bytes = 0},
        {.offset_bytes = base + tile_elem_index(dst_row, 0) * esz});
    noc.async_read(
        acc,
        dfb,
        seg,
        {.page_id = page, .offset_bytes = entry / 4},
        {.offset_bytes = base + tile_elem_index(dst_row, 16) * esz});
}

// Copy row `src_row` of source tile `page` into row `dst_row` of tile `dst_tile` (two face-row segments). Source and
// destination rows must have the same parity so both segments keep their 64 B alignment class.
template <typename Accessor>
inline void pack_row_from_row(
    const Accessor& acc,
    DataflowBuffer& dfb,
    Noc& noc,
    uint32_t page,
    uint32_t src_row,
    uint32_t dst_tile,
    uint32_t dst_row) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t seg = entry / 64;    // one face row: 32 B for bf16
    const uint32_t esz = entry / 1024;  // bytes per element
    const uint32_t base = dst_tile * entry;
    noc.async_read(
        acc,
        dfb,
        seg,
        {.page_id = page, .offset_bytes = tile_elem_index(src_row, 0) * esz},
        {.offset_bytes = base + tile_elem_index(dst_row, 0) * esz});
    noc.async_read(
        acc,
        dfb,
        seg,
        {.page_id = page, .offset_bytes = tile_elem_index(src_row, 16) * esz},
        {.offset_bytes = base + tile_elem_index(dst_row, 16) * esz});
}

// Build the packed [q | k | v] tile of value head h for user row b into tile `dst_tile` (zeroed first): channel chunk c
// goes to row 2c + (b & 1) (same parity as the source row -> 64 B aligned segments).
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void pack_head_tile_user(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h, uint32_t b, uint32_t dst_tile) {
    const uint32_t par = b & 1u;
    for (uint32_t c = 0; c < Kt; ++c) {
        pack_row_from_row(acc, dfb, noc, hk * Kt + c, b, dst_tile, 2 * c + par);
        pack_row_from_row(acc, dfb, noc, Nk * Kt + hk * Kt + c, b, dst_tile, 2 * (Kt + c) + par);
    }
    for (uint32_t c = 0; c < Vt; ++c) {
        pack_row_from_row(acc, dfb, noc, 2 * Nk * Kt + h * Vt + c, b, dst_tile, 2 * (2 * Kt + c) + par);
    }
}

// B=1 convenience (user row 0): chunk c in row 2c.
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void pack_head_tile(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h, uint32_t dst_tile) {
    pack_head_tile_user<Kt, Vt, Nk>(acc, dfb, noc, hk, h, 0, dst_tile);
}

// Selector tiles for user row b: sel[c] has a single 1.0 at (row b, col 2c + parity(b)), so sel[c] @ P puts packed row
// 2c + parity(b) of P into row b. Also the row mask e_b (1.0 at (row b, col 0)).
// The 1.0 entries of the selector / mask tiles (slots reserved and already zero-filled by the caller).
inline void set_user_selector_bits(DataflowBuffer& sel, DataflowBuffer& mask, uint32_t b, uint32_t Ct) {
    {
        auto lock = sel.scoped_write_lock(Ct);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t c = 0; c < Ct; ++c) {
            p16[c * 1024 + tile_elem_index(b, 2 * c + (b & 1u))] = 0x3F80;
        }
    }
    {
        auto lock = mask.scoped_write_lock(1);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        p16[tile_elem_index(b, 0)] = 0x3F80;
    }
}

inline void build_user_selectors(DataflowBuffer& sel, DataflowBuffer& mask, Noc& noc, uint32_t b, uint32_t Ct) {
    sel.reserve_back(Ct);
    zero_reserved(sel, noc, Ct);
    mask.reserve_back(1);
    zero_reserved(mask, noc, 1);
    set_user_selector_bits(sel, mask, b, Ct);
    sel.push_back(Ct);
    mask.push_back(1);
}

// Write rows [r0, r0 + nr) of `count` consecutive L1 tiles to the same rows of the destination tiles, exactly (no
// rounding to an even row count: with one user per core -- group size 1, B <= 9 -- the next row belongs to another
// core). Rows are written as face-row spans (32 B per row per face half); the L1 source and DRAM destination offsets
// are equal, so every transfer keeps its 64 B alignment class from any start row, the rule the readers' odd-row
// segment reads (pack_row_from_row) rely on.
template <typename Accessor>
inline void write_rows(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count, uint32_t r0, uint32_t nr) {
    dfb.wait_front(count);
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t esz = entry / 1024;
    const uint32_t seg = entry / 64;  // one face row
    const uint32_t r1 = r0 + nr;
    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t base = t * entry;
        // rows below 16 live in faces 0/1, rows >= 16 in faces 2/3; write each face span separately
        for (uint32_t lo = r0; lo < r1;) {
            const uint32_t hi = (lo < 16) ? (r1 < 16 ? r1 : 16) : r1;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = tile_elem_index(lo, half * 16) * esz;
                noc.async_write(
                    dfb,
                    acc,
                    seg * (hi - lo),
                    {.offset_bytes = base + off},
                    {.page_id = first_page + t, .offset_bytes = off});
            }
            lo = hi;
        }
    }
    noc.async_write_barrier();
    dfb.pop_front(count);
}

// ---- multi-token (speculative-verify) mode
// ---------------------------------------------------------------------------

// Rows [lo, hi) of `count` consecutive tiles at pages first_page.. -> the same rows of the reserved L1 tiles of `dfb`
// (from its write pointer). lo must be even so every face-row span starts 64 B aligned. No barrier / push.
template <typename Accessor>
inline void read_rows_span(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count, uint32_t lo, uint32_t hi) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t esz = entry / 1024;
    const uint32_t seg = entry / 64;  // one face row
    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t base = t * entry;
        for (uint32_t l = lo; l < hi;) {
            const uint32_t h = (l < 16) ? (hi < 16 ? hi : 16) : hi;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = tile_elem_index(l, half * 16) * esz;
                noc.async_read(
                    acc,
                    dfb,
                    seg * (h - l),
                    {.page_id = first_page + t, .offset_bytes = off},
                    {.offset_bytes = base + off});
            }
            l = h;
        }
    }
}

// Rows [lo, hi) of the `count` front tiles of `dfb` (read pointer; tile 0 for every destination tile when `same_src`)
// -> the same rows of the tiles at first_page... lo even. No wait / barrier / pop (unlike write_rows).
template <typename Accessor>
inline void write_rows_span(
    const Accessor& acc,
    DataflowBuffer& dfb,
    Noc& noc,
    uint32_t first_page,
    uint32_t count,
    uint32_t lo,
    uint32_t hi,
    bool same_src = false) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t esz = entry / 1024;
    const uint32_t seg = entry / 64;
    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t base = same_src ? 0u : t * entry;
        for (uint32_t l = lo; l < hi;) {
            const uint32_t h = (l < 16) ? (hi < 16 ? hi : 16) : hi;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = tile_elem_index(l, half * 16) * esz;
                noc.async_write(
                    dfb,
                    acc,
                    seg * (h - l),
                    {.offset_bytes = base + off},
                    {.page_id = first_page + t, .offset_bytes = off});
            }
            l = h;
        }
    }
}

// Selector tiles of one token: sel[c] = 1.0 at (row d, col 2c + par) so sel[c] @ P moves packed row 2c + par of P into
// row d; mask = e_d. Slots reserved and zero-filled by the caller. (T=1: d = b, par = b & 1 -> set_user_selector_bits.)
inline void set_token_selector_bits(DataflowBuffer& sel, DataflowBuffer& mask, uint32_t d, uint32_t par, uint32_t Ct) {
    {
        auto lock = sel.scoped_write_lock(Ct);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t c = 0; c < Ct; ++c) {
            p16[c * 1024 + tile_elem_index(d, 2 * c + par)] = 0x3F80;
        }
    }
    {
        auto lock = mask.scoped_write_lock(1);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        p16[tile_elem_index(d, 0)] = 0x3F80;
    }
}

// Pack row `src_row` (0..31) of the head's [q | k | v] tiles of the tile row whose first page is `page_base` into the
// reserved, zero-filled tile `dst_tile` of `dfb` at parity `par` (chunk c -> row 2c + par). A DRAM -> L1 face-row
// segment must keep its 64 B alignment class (= the row parity), so a source row of the other parity is first packed
// at its own parity into staging tile `src_row & 1` of `stage` (two reserved tiles whose other-parity rows stay zero)
// and then moved by four local L1 -> L1 face-span copies: in a face the chunk rows sit at a 64 B pitch, so one span
// per face carries all of them together with the zero rows in between. Pure data movement (bit-exact); ends with a
// read barrier.
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void pack_head_tile_row_parity(
    const Accessor& acc,
    DataflowBuffer& dfb,
    uint32_t dst_tile,
    DataflowBuffer& stage,
    Noc& noc,
    uint32_t hk,
    uint32_t h,
    uint32_t page_base,
    uint32_t src_row,
    uint32_t par) {
    constexpr uint32_t Ct = 2 * Kt + Vt;
    const uint32_t p = src_row & 1u;
    auto pack_into = [&](DataflowBuffer& d, uint32_t tile, uint32_t q) {
        for (uint32_t c = 0; c < Kt; ++c) {
            pack_row_from_row(acc, d, noc, page_base + hk * Kt + c, src_row, tile, 2 * c + q);
            pack_row_from_row(acc, d, noc, page_base + Nk * Kt + hk * Kt + c, src_row, tile, 2 * (Kt + c) + q);
        }
        for (uint32_t c = 0; c < Vt; ++c) {
            pack_row_from_row(acc, d, noc, page_base + 2 * Nk * Kt + h * Vt + c, src_row, tile, 2 * (2 * Kt + c) + q);
        }
    };
    if (p == par) {
        pack_into(dfb, dst_tile, par);
        noc.async_read_barrier();
        return;
    }
    pack_into(stage, p, p);
    noc.async_read_barrier();
    const uint32_t entry = dfb.get_entry_size();  // bf16 tile: face = entry / 4, face row 32 B, chunk pitch 64 B
    const uint32_t face = entry / 4;
    const uint32_t src = stage.get_write_ptr() + p * entry + 32 * p;
    const uint32_t dst = dfb.get_write_ptr() + dst_tile * entry + 32 * par;
    constexpr uint32_t c_lo = Ct < 8 ? Ct : 8;      // chunk rows in faces 0 / 1 (tile rows 0..15)
    constexpr uint32_t c_hi = Ct > 8 ? Ct - 8 : 0;  // chunk rows in faces 2 / 3
    const uint8_t noc_id = noc.get_noc_id();
    for (uint32_t f = 0; f < 2; ++f) {
        noc_async_read(get_noc_addr(src + f * face, noc_id), dst + f * face, (c_lo - 1) * 64 + 32, noc_id);
    }
    if constexpr (c_hi > 0) {
        for (uint32_t f = 2; f < 4; ++f) {
            noc_async_read(get_noc_addr(src + f * face, noc_id), dst + f * face, (c_hi - 1) * 64 + 32, noc_id);
        }
    }
    noc.async_read_barrier();
}

// The group's accept counts: page 0 of the ROW_MAJOR accept tensor (ACC_BYTES = B*4 rounded up to 64 B, within the
// buffer's aligned page) into the reserved slot of `dfb`, then out[i] = min(accept[u0 + i], T - 1) for i < nu
// (uint32 words; an INT32 negative clamps to T - 1 as well).
template <uint32_t ACC_BYTES, uint32_t T, typename Accessor>
inline void load_accept(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t u0, uint32_t nu, uint32_t* out) {
    noc.async_read(acc, dfb, ACC_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    auto lock = dfb.scoped_write_lock(1);
    auto p = lock.template get_ptr<volatile uint32_t>();
    for (uint32_t i = 0; i < nu; ++i) {
        const uint32_t v = p[u0 + i];
        out[i] = v > T - 1 ? T - 1 : v;
    }
}

}  // namespace gdn_step_df
