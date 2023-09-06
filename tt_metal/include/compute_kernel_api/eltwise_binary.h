/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB.h"
#endif



namespace ckernel {

ALWI void binary_op_init_common(uint32_t icb0, uint32_t icb1)
{
    UNPACK(( llk_setup_operands() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<BroadcastType::NONE>(icb0, icb1) ));
    #else
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1) ));
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated(icb0, icb1) ));
    #endif

    MATH(( llk_math_pack_sync_init<SYNC>() ));


    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>() ));
}


ALWI void mul_tiles_init_f() { MATH(( llk_math_eltwise_binary_init<ELWMUL, NONE, MATH_FIDELITY>() )); }
ALWI void mul_tiles_init() {
    MATH(( llk_math_eltwise_binary_init<ELWMUL, NONE, MATH_FIDELITY>() ));
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
    #else
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(0, 1) ));
    #endif
}

ALWI void add_tiles_init_nof() { MATH(( llk_math_eltwise_binary_init<ELWADD, NONE>() )); }
ALWI void add_tiles_init() {
    MATH(( llk_math_eltwise_binary_init<ELWADD, NONE>() ));
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
    #else
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(0, 1) ));
    #endif
}

ALWI void sub_tiles_init_nof() { MATH(( llk_math_eltwise_binary_init<ELWSUB, NONE>() )); }
ALWI void sub_tiles_init() {
    MATH(( llk_math_eltwise_binary_init<ELWSUB, NONE>() ));
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
    #else
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(0, 1) ));
    #endif
}


/**
 * Performs element-wise multiplication C=A*B of tiles in two CBs at given
 * indices and writes the result to the DST register at index dst_tile_index.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
ALWI void mul_tiles( uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    //static bool first = true; // TODO(AP): static initializer causes a hang, possibly investigate
    //if (first)
    // one possible solution is to add a local context in the kernel, pass it around and store init flags in it
    // this way the compiler should be able to perform loop hoisting optimization
    // - might need to add __attribute__((pure)) to init calls for this to work
    // Also pass -fmove-loop-invariants to g++
    //mul_tiles_initf();
    //first = false;

    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1)  ));
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_binary<ELWMUL, NONE, SyncHalf, MATH_FIDELITY, false>(idst) ));
    #else
    MATH(( llk_math_eltwise_binary<ELWMUL, NONE, SyncHalf, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst) ));
    #endif
}

/**
 * Performs element-wise addition C=A+B of tiles in two CBs at given indices
 * and writes the result to the DST register at index dst_tile_index. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call
 * is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
ALWI void add_tiles( uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1)  ));

    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_binary<ELWADD, NONE, SyncHalf, MATH_FIDELITY, false>(idst) ));
    #else
    MATH(( llk_math_eltwise_binary<ELWADD, NONE, SyncHalf, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst) ));
    #endif
}

/**
 * Performs element-wise subtraction C=A-B of tiles in two CBs at given indices
 * and writes the result to the DST register at index dst_tile_index. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call
 * is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
ALWI void sub_tiles( uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1)  ));

    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_binary<ELWSUB, NONE, SyncHalf, MATH_FIDELITY, false>(idst) ));
    #else
    MATH(( llk_math_eltwise_binary<ELWSUB, NONE, SyncHalf, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst) ));
    #endif
}

ALWI void binary_op_specific_init(int op_code) // TODO(AP): better naming
{
    #ifdef ELTWISE_OP
    if constexpr (ELTWISE_OP_CODE == 0) // TODO(AP): pass an enum probably
        add_tiles_init_nof();
    if constexpr (ELTWISE_OP_CODE == 1)
        sub_tiles_init_nof();
    if constexpr (ELTWISE_OP_CODE == 2)
        mul_tiles_init_f();
    #endif
}


} // namespace ckernel
