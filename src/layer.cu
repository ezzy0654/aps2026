#include <cstring>
#include "layer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

Linear::Linear(const ModelLoader& loader, const std::string& weight, const std::string& bias)
    : weight_(loader.load(weight)) {
    if (!bias.empty() && loader.has(bias)) bias_ = loader.load(bias);
    // Weights never change after this; keep a persistent device copy so
    // every forward() call reuses it instead of re-uploading it each time.
    weight_.to_device();
    if (bias_.size()) bias_.to_device();
}

void Linear::forward(const Tensor& x, Tensor& y) const {
    const std::size_t cols = x.size(x.ndim() - 1);
    const std::size_t rows = x.size() / cols;
    y = Tensor({rows, weight_.size(0)});
    if (x.ndim() == 2) {
        // Already [rows, cols] — the reshape below would be a no-op, so skip
        // the copy entirely.
        tensor_ops::matmul_transposed(x, weight_, y);
    } else {
        Tensor flat = x;
        flat.reshape({rows, cols});
        tensor_ops::matmul_transposed(flat, weight_, y);
    }
    if (bias_.size()) tensor_ops::add_bias_inplace(y, bias_);
}

void Linear::forward_device(const float* d_x, float* d_y, std::size_t rows) const {
    // Bias rides along in the GEMM epilogue rather than in a second kernel.
    tensor_ops::device::matmul_transposed(d_x, weight_, d_y, rows,
                                          bias_.size() ? &bias_ : nullptr);
}

PhiMLP::PhiMLP(const ModelLoader& loader, const std::string& prefix)
    : w1_(loader.load(prefix + ".w1.weight")),
      w2_(loader.load(prefix + ".w2.weight")),
      w3_(loader.load(prefix + ".w3.weight")) {
    w1_.to_device();
    w2_.to_device();
    w3_.to_device();
}

void PhiMLP::forward_device(const float* d_x, float* d_y, std::size_t rows) const {
    APS_PROFILE_SCOPE("mlp.forward");
    const std::size_t inter = w1_.size(0);
    float* d_gate = tensor_ops::device::buffer(tensor_ops::device::Buffer::Gate, rows * inter);
    float* d_up = tensor_ops::device::buffer(tensor_ops::device::Buffer::Up, rows * inter);
    { APS_PROFILE_SCOPE("mm.w1      [c,4096]x[448,4096]"); tensor_ops::device::matmul_transposed(d_x, w1_, d_gate, rows); }
    { APS_PROFILE_SCOPE("mlp.silu");                        tensor_ops::device::silu(d_gate, d_gate, rows * inter); }
    { APS_PROFILE_SCOPE("mm.w3      [c,4096]x[448,4096]"); tensor_ops::device::matmul_transposed(d_x, w3_, d_up, rows); }
    { APS_PROFILE_SCOPE("mlp.mul");                         tensor_ops::device::mul(d_gate, d_up, d_gate, rows * inter); }
    { APS_PROFILE_SCOPE("mm.w2      [c,448]x[4096,448]");  tensor_ops::device::matmul_transposed(d_gate, w2_, d_y, rows); }
}

// The reference selection, kept as the specification this model's routing has
// to match: device::route_top2 is a transliteration of the two scans below and
// is what actually runs (see PhiMoE::forward_device). Retained because it is
// the oracle any change to the device kernel must be diffed against -- the
// arithmetic here is load-bearing in a way no comment can substitute for.
//
// `logits` points at this token's NUM_EXPERTS gate scores. Everything here
// is a fixed 16 wide, so it lives on the stack: the previous version built a
// Tensor (and two vectors) per token, which cost ~630k heap allocations per
// chunk for 64 bytes of data. The quantise-then-select arithmetic below is
// untouched -- it decides which expert a token goes to, and any change to it
// risks the routing flips that cost a session to diagnose (see
// apss26-wiki/wiki/matching-reference-numerics.md).
void PhiMoE::route(const float* logits, std::pair<int, float> routes[2]) const {
    float scores[apss26::NUM_EXPERTS];
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
        const float score = logits[e];
        const float rounded = std::floor(std::fabs(score) /
            apss26::ROUTER_SCORE_QUANTUM + 0.5f) * apss26::ROUTER_SCORE_QUANTUM;
        scores[e] = score < 0.0f ? -rounded : rounded;
    }
    auto select = [&scores](int excluded) {
        int best = -1; float best_value = -std::numeric_limits<float>::infinity();
        for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
            if (static_cast<int>(e) == excluded) continue;
            if (best < 0 || scores[e] > best_value + apss26::ROUTER_TIE_EPS) {
                best = static_cast<int>(e); best_value = scores[e];
            }
        }
        return best;
    };
    const int first = select(-1);
    const int second = select(first);
    routes[0] = {first, 0.5f};
    routes[1] = {second, 0.5f};
}

PhiMoE::PhiMoE(const ModelLoader& loader, std::size_t layer_idx) {
    const std::string base = "model.layers." + std::to_string(layer_idx) + ".block_sparse_moe";
    gate_ = loader.load(base + ".gate.weight");
    gate_.to_device();

    // Stack the 16 experts' weights into one tensor each, expert e at rows
    // [e*N, (e+1)*N). Same bytes, same values -- only the layout changes, and
    // it happens before the timer starts. What it buys is that all 16 expert
    // FFNs become a single grouped launch (see forward_device below).
    //
    // w1 and w3 are additionally concatenated along N (see layer.h): expert
    // e's block of w13_all_ holds its w1 rows [e*896, e*896+448) then its w3
    // rows [e*896+448, (e+1)*896).
    const std::string first = base + ".experts.0";
    const Tensor w1_0 = loader.load(first + ".w1.weight");
    const Tensor w2_0 = loader.load(first + ".w2.weight");
    const Tensor w3_0 = loader.load(first + ".w3.weight");
    const std::size_t inter = w1_0.size(0), hidden = w1_0.size(1);
    const std::size_t E = apss26::NUM_EXPERTS;
    w13_all_ = Tensor({E * 2 * inter, hidden});
    w2_all_ = Tensor({E * hidden, inter});

    const auto stack_rows = [](Tensor& dst, const Tensor& src, std::size_t row_offset, std::size_t row_width) {
        std::memcpy(dst.data() + row_offset * row_width, src.data(), src.size() * sizeof(float));
    };
    for (std::size_t e = 0; e < E; ++e) {
        const std::string prefix = base + ".experts." + std::to_string(e);
        const Tensor w1_e = (e == 0) ? w1_0 : loader.load(prefix + ".w1.weight");
        const Tensor w2_e = (e == 0) ? w2_0 : loader.load(prefix + ".w2.weight");
        const Tensor w3_e = (e == 0) ? w3_0 : loader.load(prefix + ".w3.weight");
        stack_rows(w13_all_, w1_e, e * 2 * inter, hidden);
        stack_rows(w13_all_, w3_e, e * 2 * inter + inter, hidden);
        stack_rows(w2_all_, w2_e, e * hidden, inter);
    }
    w13_all_.to_device();
    w2_all_.to_device();
}

// Tile height for the three expert GEMMs. 64 was measured first (it is what
// the per-expert path used): fusing alone took w1/w3 from 29% to 45% of peak.
// 128 doubles arithmetic intensity, 16 -> 32 flop/byte, at the cost of padding
// N=448 up to 512 -- 12.5% of the w1/w3 columns wasted. That trade is only
// available because fusion keeps the block count high; at 128 a single expert
// would yield 40 blocks against 82 SMs, which is exactly what made v9 reject
// the larger tile here.
namespace { constexpr std::size_t kExpertBlockM = 128; }

void PhiMoE::forward_device(const float* d_x, float* d_y, std::size_t rows, std::size_t h) const {
    APS_PROFILE_SCOPE("moe.forward");
    namespace device = tensor_ops::device;
    const std::size_t E = apss26::NUM_EXPERTS;
    const std::size_t inter = w13_all_.size(0) / (2 * E);

    // The hidden state is already device-resident; it is both the gate
    // matmul's input and every expert's gather source.
    const float* d_flat = d_x;

    float* d_router = device::buffer(device::Buffer::Router, rows * E);
    {
        APS_PROFILE_SCOPE("moe.gate_matmul");
        device::matmul_transposed(d_flat, gate_, d_router, rows);
    }

    std::vector<std::vector<std::pair<std::size_t, float>>> assignments(E);
    {
        APS_PROFILE_SCOPE("moe.route");
        // Selection runs on the device (see device::route_top2); what the host
        // still does is bucket the rows by expert. Reused across layers and
        // chunks so this is not 32 allocations of the same 125 KB -- the host
        // is single-threaded here, and forward_device is only ever called from
        // the one decoder-layer loop.
        static std::vector<int> pairs;
        pairs.resize(2 * rows);
        device::route_top2(d_router, pairs.data(), rows);
        for (std::size_t t = 0; t < rows; ++t) {
            // Both weights are the model's fixed 0.5; expert order within a
            // row is (first, second), which is what fixes the accumulation
            // order in the per-expert scatter below.
            assignments[pairs[2 * t]].emplace_back(t, 0.5f);
            assignments[pairs[2 * t + 1]].emplace_back(t, 0.5f);
        }
    }

    // Lay every expert's rows out back to back in one buffer, expert 0 first.
    // That single layout is what lets gather, w1, silu, w3, mul and w2 each
    // run as ONE launch over all 16 experts instead of 16 serialised ones --
    // an expert alone yields ~220 blocks against the ~490 this GPU holds.
    std::vector<int> all_idx;
    std::vector<float> all_w;
    std::vector<std::size_t> off(E + 1, 0);
    std::vector<device::GroupTile> tiles;
    // scatter_pos[2*t] / scatter_pos[2*t+1]: the two d_out rows (see below)
    // holding token t's two expert results, filled in as each token's
    // assignment is flattened -- the "which of the two free slots" tracked
    // by `slot_used`. This is what lets the final combine skip a device
    // bucketing pass: it already knows, from this host loop, exactly where
    // in d_out each token's pair landed.
    static std::vector<int> scatter_pos;
    static std::vector<unsigned char> slot_used;
    scatter_pos.assign(2 * rows, -1);
    slot_used.assign(rows, 0);
    {
        APS_PROFILE_SCOPE("moe.expert_setup");
        all_idx.reserve(2 * rows);
        all_w.reserve(2 * rows);
        for (std::size_t e = 0; e < E; ++e) {
            off[e] = all_idx.size();
            for (const auto& a : assignments[e]) {
                const std::size_t t = a.first;
                scatter_pos[2 * t + slot_used[t]] = static_cast<int>(all_idx.size());
                ++slot_used[t];
                all_idx.push_back(static_cast<int>(t));
                all_w.push_back(a.second);
            }
        }
        off[E] = all_idx.size();

        // A tile never straddles an expert boundary, so no block reads another
        // expert's rows. The map depends only on the tile height, so with all
        // three matmuls on the same height one map serves them all.
        for (std::size_t e = 0; e < E; ++e)
            for (std::size_t r = off[e]; r < off[e + 1]; r += kExpertBlockM)
                tiles.push_back({static_cast<int>(r), static_cast<int>(off[e + 1]),
                                 static_cast<int>(e)});
    }
    const std::size_t total = all_idx.size();
    if (total == 0) {
        device::check(cudaMemset(d_y, 0, rows * h * sizeof(float)), "moe y zero");
        return;
    }

    int* d_idx = device::index_buffer(total);
    float* d_w = device::weight_buffer(total);
    const device::GroupTile* d_tiles = device::upload_group_tiles(tiles.data(), tiles.size());
    device::check(cudaMemcpy(d_idx, all_idx.data(), total * sizeof(int),
                             cudaMemcpyHostToDevice), "moe indices H2D");
    device::check(cudaMemcpy(d_w, all_w.data(), total * sizeof(float),
                             cudaMemcpyHostToDevice), "moe weights H2D");

    float* d_in = device::buffer(device::Buffer::Input, total * h);
    float* d_out = device::buffer(device::Buffer::Output, total * h);
    float* d_gateup = device::buffer(device::Buffer::Up, total * 2 * inter);
    float* d_gate = device::buffer(device::Buffer::Gate, total * inter);

    {
        APS_PROFILE_SCOPE("moe.gather");
        device::gather_rows(d_flat, d_in, d_idx, total, h);
    }
    {
        APS_PROFILE_SCOPE("mlp.forward");
        // w1 and w3 run as one 896-wide grouped GEMM (see layer.h): 448+448
        // is exactly 7*128, so the 128-wide tile needs no padding, where 448
        // alone pads to 512 (12.5% wasted columns). silu_mul_fused folds the
        // silu+mul pair into one pass over the result.
        { APS_PROFILE_SCOPE("mm.w13     [c,4096]x[896,4096]");
          device::matmul_transposed_grouped(d_in, w13_all_, E, d_gateup, d_tiles,
                                            tiles.size(), kExpertBlockM); }
        { APS_PROFILE_SCOPE("mlp.silu_mul");
          device::silu_mul_fused(d_gateup, d_gate, total, inter); }
        { APS_PROFILE_SCOPE("mm.w2      [c,448]x[4096,448]");
          device::matmul_transposed_grouped(d_gate, w2_all_, E, d_out, d_tiles,
                                            tiles.size(), kExpertBlockM); }
    }

    {
        // Every token has exactly two entries in scatter_pos (its two expert
        // picks), both weighted 0.5 -- an exact power of two -- so the sum is
        // order-independent (see the kernel comment in tensor.cu) and this
        // needs neither the up-front zero-fill nor per-expert launches: one
        // pass over d_y's rows reads both of a token's positions directly.
        APS_PROFILE_SCOPE("moe.scatter");
        device::scatter_pairs(d_out, scatter_pos.data(), d_y, rows, h);
    }
}

PhiAttention::PhiAttention(const ModelLoader& loader, std::size_t layer_idx)
    : q_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.bias"),
      k_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.bias"),
      v_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.bias"),
      o_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.bias") {}

void PhiAttention::forward_device(const float* d_x, float* d_y, std::size_t rows,
                                  const tensor_ops::device::RowMap& row_map,
                                  const float2* rope_table) const {
    APS_PROFILE_SCOPE("attention.forward");
    namespace device = tensor_ops::device;
    constexpr std::size_t q_dim = apss26::NUM_ATTENTION_HEADS * apss26::HEAD_DIM;
    constexpr std::size_t kv_dim = apss26::NUM_KV_HEADS * apss26::HEAD_DIM;

    float* d_q = device::buffer(device::Buffer::Q, rows * q_dim);
    float* d_k = device::buffer(device::Buffer::K, rows * kv_dim);
    float* d_v = device::buffer(device::Buffer::V, rows * kv_dim);
    {
        APS_PROFILE_SCOPE("attention.qkv_proj");
        { APS_PROFILE_SCOPE("mm.q_proj  [T,4096]x[2048,4096]"); q_proj_.forward_device(d_x, d_q, rows); }
        { APS_PROFILE_SCOPE("mm.k_proj  [T,4096]x[512,4096]");  k_proj_.forward_device(d_x, d_k, rows); }
        { APS_PROFILE_SCOPE("mm.v_proj  [T,4096]x[512,4096]");  v_proj_.forward_device(d_x, d_v, rows); }
    }
    {
        APS_PROFILE_SCOPE("attention.rope");
        // theta is baked into rope_table, built once per chunk in forward_chunk.
        device::apply_rope(d_q, d_k, rows, apss26::NUM_ATTENTION_HEADS, apss26::NUM_KV_HEADS,
                           apss26::HEAD_DIM, row_map, rope_table);
    }
    float* d_out = device::buffer(device::Buffer::AttnOut, rows * q_dim);
    {
        APS_PROFILE_SCOPE("attention.core");
        device::sliding_window_attention(d_q, d_k, d_v, rows, apss26::NUM_ATTENTION_HEADS,
                                         apss26::NUM_KV_HEADS, apss26::HEAD_DIM,
                                         apss26::SLIDING_WINDOW, row_map, d_out);
    }
    {
        APS_PROFILE_SCOPE("attention.o_proj");
        { APS_PROFILE_SCOPE("mm.o_proj  [T,2048]x[4096,2048]"); o_proj_.forward_device(d_out, d_y, rows); }
    }
}

PhiDecoderLayer::PhiDecoderLayer(const ModelLoader& loader, std::size_t layer_idx)
    : input_norm_weight_(loader.load("model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight")),
      input_norm_bias_(loader.load("model.layers." + std::to_string(layer_idx) + ".input_layernorm.bias")),
      post_norm_weight_(loader.load("model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight")),
      post_norm_bias_(loader.load("model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.bias")),
      attention_(loader, layer_idx), moe_(loader, layer_idx) {
    input_norm_weight_.to_device();
    input_norm_bias_.to_device();
    post_norm_weight_.to_device();
    post_norm_bias_.to_device();
}

void PhiDecoderLayer::forward_device(float* d_x, const float* d_carry, float* d_y,
                                     std::size_t rows,
                                     const tensor_ops::device::RowMap& row_map,
                                     const float2* rope_table) const {
    APS_PROFILE_SCOPE("decoder.layer.forward");
    namespace device = tensor_ops::device;
    const std::size_t h = apss26::HIDDEN_SIZE;
    const std::size_t n = rows * h;

    float* d_normed = device::buffer(device::Buffer::Normed, n);
    if (d_carry != nullptr) {
        // The previous layer's second residual, deferred to here so this
        // LayerNorm's staging pass absorbs it instead of a separate kernel
        // writing a buffer this one would immediately re-read. d_carry points
        // at Buffer::Attn, which attention_ below overwrites -- safe because
        // this read completes first on the same stream.
        APS_PROFILE_SCOPE("layer.norm_add");
        device::add_layer_norm(d_x, d_carry, input_norm_weight_, input_norm_bias_,
                               apss26::NORM_EPS, d_normed, rows, h);
    } else {
        APS_PROFILE_SCOPE("layer.norm");
        device::layer_norm(d_x, input_norm_weight_, input_norm_bias_, apss26::NORM_EPS,
                           d_normed, rows, h);
    }

    float* d_attn = device::buffer(device::Buffer::Attn, n);
    attention_.forward_device(d_normed, d_attn, rows, row_map, rope_table);

    float* d_post = device::buffer(device::Buffer::Post, n);
    { APS_PROFILE_SCOPE("layer.norm_add");
      // The attention residual used to be its own add_inplace kernel and this
      // LayerNorm then re-read its output -- one whole extra pass over the
      // residual stream. Folded into the norm's staging pass; d_attn still
      // ends up holding attn + x, which the second residual below needs.
      device::add_layer_norm(d_attn, d_x, post_norm_weight_, post_norm_bias_,
                             apss26::NORM_EPS, d_post, rows, h); }

    // The second residual is NOT closed here: d_y stays the raw MoE output and
    // d_attn stays live as the carry. The next layer's input LayerNorm adds
    // them during a pass it was going to make anyway.
    moe_.forward_device(d_post, d_y, rows, h);
}
