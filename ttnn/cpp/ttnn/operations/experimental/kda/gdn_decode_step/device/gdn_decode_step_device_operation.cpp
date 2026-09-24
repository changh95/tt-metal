// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "gdn_decode_step_device_operation.hpp"

#include <array>
#include <optional>
#include <cmath>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
constexpr std::string_view kOp = "gdn_decode_step";

void check_tiled(const Tensor& t, std::string_view name, std::initializer_list<DataType> dtypes) {
    using namespace kda_factory_detail;
    check_allocated_device_tensor(t, kOp, name);
    check_layout(t, Layout::TILE, kOp, name);
    check_interleaved(t, kOp, name);
    bool ok = false;
    for (auto d : dtypes) {
        ok = ok || t.dtype() == d;
    }
    TT_FATAL(ok, "{}: {} has unsupported dtype {}", kOp, name, t.dtype());
}
}  // namespace

GdnDecodeStepOperation::program_factory_t GdnDecodeStepOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GdnDecodeStepProgramFactory{};
}

void GdnDecodeStepOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& in) {
    using namespace kda_factory_detail;
    check_tiled(in.qkv, "qkv", {DataType::BFLOAT16});
    check_tiled(in.beta, a.fuse_conv ? "dt_bias" : "beta", {DataType::FLOAT32, DataType::BFLOAT16});
    check_tiled(in.g, a.fuse_conv ? "neg_exp_A" : "g", {DataType::FLOAT32, DataType::BFLOAT16});
    check_tiled(in.state, "state", {DataType::FLOAT32});
    check_tiled(in.weight, "weight", {DataType::BFLOAT16});
    for (const auto& [t, name] : std::array{
             std::pair{&in.beta, "beta"},
             std::pair{&in.g, "g"},
             std::pair{&in.state, "state"},
             std::pair{&in.weight, "weight"}}) {
        check_same_device(in.qkv, *t, kOp, name);
    }
    check_output_interleaved(a.output_mem_config, kOp);
    check_compute_config(a.compute_kernel_config, kOp);
    TT_FATAL(
        a.compute_kernel_config.fp32_dest_acc_en,
        "{}: fp32_dest_acc_en must be enabled (32-bit state accumulation and in-DST transpose)",
        kOp);
    TT_FATAL(!a.compute_kernel_config.packer_l1_acc, "{}: packer_l1_acc is unsupported", kOp);
    TT_FATAL(
        a.output_dtype == DataType::FLOAT32 || a.output_dtype == DataType::BFLOAT16,
        "{}: output_dtype must be FLOAT32 or BFLOAT16",
        kOp);
    const uint32_t Nv = a.num_value_heads, Nk = a.num_key_heads, Dk = a.key_dim, Dv = a.value_dim;
    TT_FATAL(Nv > 0 && Nk > 0 && Nv % Nk == 0, "{}: num_value_heads must be a multiple of num_key_heads", kOp);
    TT_FATAL(Nv <= tt::constants::TILE_WIDTH, "{}: num_value_heads must fit one tile row (<= 32)", kOp);
    TT_FATAL(
        Dk > 0 && Dv > 0 && Dk % tt::constants::TILE_WIDTH == 0 && Dv % tt::constants::TILE_WIDTH == 0,
        "{}: key_dim and value_dim must be tile aligned",
        kOp);
    TT_FATAL(std::isfinite(a.scale) && a.l2_epsilon > 0.0f && a.norm_epsilon > 0.0f, "{}: bad scale/epsilon", kOp);
    const auto& qs = in.qkv.logical_shape();
    const uint32_t C = 2 * Nk * Dk + Nv * Dv;
    const bool mt = a.num_tokens > 1;
    if (mt) {
        TT_FATAL(a.fuse_conv, "{}: num_tokens > 1 (multi-token verify mode) needs the fused-conv mode", kOp);
        TT_FATAL(
            in.qkv_prev.has_value() && in.accept.has_value(),
            "{}: num_tokens > 1 needs qkv_prev (previous step's projection rows) and accept",
            kOp);
        TT_FATAL(
            a.num_tokens <= tt::constants::TILE_HEIGHT, "{}: num_tokens must be <= 32 (got {})", kOp, a.num_tokens);
        const auto& acc = *in.accept;
        check_allocated_device_tensor(acc, kOp, "accept");
        check_layout(acc, Layout::ROW_MAJOR, kOp, "accept");
        check_interleaved(acc, kOp, "accept");
        check_same_device(in.qkv, acc, kOp, "accept");
        TT_FATAL(
            acc.dtype() == DataType::UINT32 || acc.dtype() == DataType::INT32,
            "{}: accept must be UINT32 or INT32 (got {})",
            kOp,
            acc.dtype());
        const uint32_t B = static_cast<uint32_t>(acc.logical_volume());
        TT_FATAL(
            B >= 1 && acc.logical_shape()[-1] == B,
            "{}: accept must be a single row [B] or [1, B] (got {})",
            kOp,
            acc.logical_shape());
        TT_FATAL(
            (qs.rank() == 4 && qs[0] == 1 && qs[1] == 1) || (qs.rank() == 3 && qs[0] == 1),
            "{}: multi-token qkv must be [1, 1, R, W] or [1, R, W] (got {})",
            kOp,
            qs);
        const uint32_t R = qs[-2];
        TT_FATAL(
            R >= B * a.num_tokens && qs[-1] >= a.qkvz_dim + 2 * Nv,
            "{}: multi-token qkv needs R >= B*T rows ({} < {}*{}) and W >= qkvz_dim + 2*Nv",
            kOp,
            R,
            B,
            a.num_tokens);
        check_tiled(*in.qkv_prev, "qkv_prev", {DataType::BFLOAT16});
        check_same_device(in.qkv, *in.qkv_prev, kOp, "qkv_prev");
        TT_FATAL(
            in.qkv_prev->logical_shape() == qs && in.qkv_prev->padded_shape() == in.qkv.padded_shape(),
            "{}: qkv_prev must have qkv's shape (got {} vs {})",
            kOp,
            in.qkv_prev->logical_shape(),
            qs);
    }
    if (a.fuse_conv) {
        TT_FATAL(
            mt || (qs.rank() == 3 && qs[0] == 1 && qs[1] >= 1 && qs[1] <= tt::constants::TILE_HEIGHT &&
                   qs[2] >= a.qkvz_dim + 2 * Nv),
            "{}: fused-conv qkv (projection rows) must be [1, B <= 32, W >= qkvz_dim + 2*Nv] (got {})",
            kOp,
            qs);
        TT_FATAL(
            a.qkvz_dim == C + Nv * Dv && a.qkvz_dim % tt::constants::TILE_WIDTH == 0 &&
                2 * Nv <= tt::constants::TILE_WIDTH,
            "{}: fused-conv needs qkvz_dim == 2*Nk*Dk + 2*Nv*Dv (tile aligned) and 2*Nv <= 32 (a|b in one tile)",
            kOp);
        TT_FATAL(
            in.conv_hist.has_value() && in.conv_taps.has_value(), "{}: fused-conv needs conv_hist and conv_taps", kOp);
        check_tiled(*in.conv_hist, "conv_hist", {DataType::BFLOAT16});
        check_tiled(*in.conv_taps, "conv_taps", {DataType::BFLOAT16});
        check_same_device(in.qkv, *in.conv_hist, kOp, "conv_hist");
        check_same_device(in.qkv, *in.conv_taps, kOp, "conv_taps");
        const auto& hs = in.conv_hist->logical_shape();
        const auto& ts = in.conv_taps->logical_shape();
        const uint32_t users = mt ? static_cast<uint32_t>(in.accept->logical_volume()) : static_cast<uint32_t>(qs[1]);
        TT_FATAL(
            hs.rank() == 5 && hs[0] >= users && hs[1] == Nv && hs[2] == 4 && hs[3] == tt::constants::TILE_HEIGHT &&
                hs[4] == tt::constants::TILE_WIDTH,
            "{}: conv_hist must be packed [Bmax >= B, Nv, 4, 32, 32] (got {})",
            kOp,
            hs);
        TT_FATAL(
            ts.rank() == 4 && ts[0] == Nv && ts[1] == 4 && ts[2] == tt::constants::TILE_HEIGHT &&
                ts[3] == tt::constants::TILE_WIDTH,
            "{}: conv_taps must be packed [Nv, 4, 32, 32] (got {})",
            kOp,
            ts);
        TT_FATAL(
            2 * ((2 * Dk + Dv) / tt::constants::TILE_WIDTH) <= tt::constants::TILE_HEIGHT,
            "{}: packed head row needs <= 16 chunks (chunk c in row 2c)",
            kOp);
        TT_FATAL(
            in.beta.logical_volume() == Nv && in.g.logical_volume() == Nv,
            "{}: dt_bias / neg_exp_A volume must be Nv",
            kOp);
    } else {
        TT_FATAL(
            qs.rank() == 3 && qs[0] == 1 && qs[1] == 1 && qs[2] == C,
            "{}: qkv must be [1, 1, 2*Nk*Dk + Nv*Dv] (got {})",
            kOp,
            qs);
        for (const auto& [t, name] : std::array{std::pair{&in.beta, "beta"}, std::pair{&in.g, "g"}}) {
            const auto& s = t->logical_shape();
            TT_FATAL(
                s.rank() == 3 && s[0] == 1 && s[1] == 1 && s[2] == Nv,
                "{}: {} must be [1, 1, Nv] (got {})",
                kOp,
                name,
                s);
        }
    }
    const auto& ss = in.state.logical_shape();
    const uint32_t rows =
        mt ? static_cast<uint32_t>(in.accept->logical_volume()) : static_cast<uint32_t>(in.qkv.logical_shape()[-2]);
    TT_FATAL(
        ss.rank() == 4 && ss[0] >= rows && (a.fuse_conv || ss[0] == 1) && ss[1] == Nv && ss[2] == Dk && ss[3] == Dv,
        "{}: state must be [Bmax >= B, Nv, Dk, Dv] (got {}, B = {})",
        kOp,
        ss,
        rows);
    TT_FATAL(in.weight.logical_volume() == Dv, "{}: weight volume must equal value_dim", kOp);
}

GdnDecodeStepOperation::spec_return_value_t GdnDecodeStepOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& in) {
    // [1, R, Nv*Dv] for the one-token op ([1, B, W] qkv); the multi-token op keeps qkv's rank ([1, 1, R, W] -> [1, 1,
    // R, Nv*Dv])
    auto shape = in.qkv.logical_shape();
    shape[-1] = a.num_value_heads * a.value_dim;
    return TensorSpec(shape, TensorLayout(a.output_dtype, PageConfig(Layout::TILE), a.output_mem_config));
}

GdnDecodeStepOperation::tensor_return_value_t GdnDecodeStepOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(a, in), in.qkv.device());
}

Tensor gdn_decode_step(
    const Tensor& qkv,
    const Tensor& beta,
    const Tensor& g,
    const Tensor& state,
    const Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    float scale,
    float l2_epsilon,
    float norm_epsilon,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    DataType output_dtype,
    const std::optional<Tensor>& conv_hist,
    const std::optional<Tensor>& conv_taps,
    uint32_t qkvz_dim,
    uint32_t num_tokens,
    const std::optional<Tensor>& qkv_prev,
    const std::optional<Tensor>& accept) {
    return ttnn::device_operation::launch<GdnDecodeStepOperation>(
        GdnDecodeStepParams{
            .num_value_heads = num_value_heads,
            .num_key_heads = num_key_heads,
            .key_dim = key_dim,
            .value_dim = value_dim,
            .scale = scale,
            .l2_epsilon = l2_epsilon,
            .norm_epsilon = norm_epsilon,
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype,
            .compute_kernel_config = compute_kernel_config,
            .fuse_conv = conv_hist.has_value(),
            .qkvz_dim = qkvz_dim,
            .num_tokens = num_tokens,
        },
        GdnDecodeStepInputs{
            .qkv = qkv,
            .beta = beta,
            .g = g,
            .state = state,
            .weight = weight,
            .conv_hist = conv_hist,
            .conv_taps = conv_taps,
            .qkv_prev = qkv_prev,
            .accept = accept});
}

}  // namespace ttnn::experimental::prim
