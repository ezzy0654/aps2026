#include "layer.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

// Match the reference CUDA/PyTorch path for nonlinear and attention exponentials.
static inline float accurate_exp(float x) {
    return static_cast<float>(std::exp(static_cast<double>(x)));
}

Linear::Linear(const ModelLoader& loader, const std::string& weight, const std::string& bias)
    : weight_(loader.load(weight)) {
    if (!bias.empty() && loader.has(bias)) bias_ = loader.load(bias);
}

void Linear::forward(const Tensor& x, Tensor& y, bool use_gpu) const {
    const std::size_t rows = x.size() / x.size(x.ndim() - 1);
    y = Tensor({rows, weight_.size(0)});
    if (use_gpu) tensor_ops::matmul_transposed_gpu(x, weight_, y);
    else tensor_ops::matmul_transposed(x, weight_, y);
    if (bias_.size()) {
        if (use_gpu) tensor_ops::add_bias_inplace_gpu(y, bias_);
        else tensor_ops::add_bias_inplace(y, bias_);
    }
}

PhiMLP::PhiMLP(const ModelLoader& loader, const std::string& prefix)
    : w1_(loader.load(prefix + ".w1.weight")),
      w2_(loader.load(prefix + ".w2.weight")),
      w3_(loader.load(prefix + ".w3.weight")) {}

void PhiMLP::forward(const Tensor& x, Tensor& y, bool use_gpu) const {
    const std::size_t rows = x.size() / x.size(x.ndim() - 1);
    const auto mm = use_gpu ? tensor_ops::matmul_transposed_gpu
                            : tensor_ops::matmul_transposed;
    Tensor activated({rows, w1_.size(0)});
    if (use_gpu) {
        tensor_ops::matmul_pair_silu_gpu(x, w1_, w3_, activated);
    } else {
        Tensor gate({rows, w1_.size(0)}), up({rows, w3_.size(0)});
        mm(x, w1_, gate);
        mm(x, w3_, up);
        tensor_ops::silu(gate, activated);
        tensor_ops::mul(activated, up, activated);
    }
    Tensor out({rows, w2_.size(0)});
    mm(activated, w2_, out);
    out.reshape(x.shape());
    y = std::move(out);
}

void PhiMoE::route(const Tensor& logits, std::vector<std::pair<int, float>>& routes) const {
    routes.clear();
    std::vector<float> scores(apss26::NUM_EXPERTS);
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
        const float score = logits[e];
        const float rounded = std::floor(std::fabs(score) /
            apss26::ROUTER_SCORE_QUANTUM + 0.5f) * apss26::ROUTER_SCORE_QUANTUM;
        scores[e] = score < 0.0f ? -rounded : rounded;
    }
    auto select = [](const std::vector<float>& values, int excluded) {
        int best = -1; float best_value = -std::numeric_limits<float>::infinity();
        for (std::size_t e = 0; e < values.size(); ++e) {
            if (static_cast<int>(e) == excluded) continue;
            if (best < 0 || values[e] > best_value + apss26::ROUTER_TIE_EPS) {
                best = static_cast<int>(e); best_value = values[e];
            }
        }
        return best;
    };
    const int first = select(scores, -1);
    const int second = select(scores, first);
    routes.emplace_back(first, 0.5f);
    routes.emplace_back(second, 0.5f);
}

PhiMoE::PhiMoE(const ModelLoader& loader, std::size_t layer_idx) {
    const std::string base = "model.layers." + std::to_string(layer_idx) + ".block_sparse_moe";
    gate_ = loader.load(base + ".gate.weight");
    experts_.reserve(apss26::NUM_EXPERTS);
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e)
        experts_.emplace_back(loader, base + ".experts." + std::to_string(e));
}

void PhiMoE::forward(const Tensor& x, Tensor& y, bool use_gpu) const {
    const std::size_t rows = x.size(0), h = x.size(1);
    Tensor router({rows, apss26::NUM_EXPERTS});
    if (use_gpu) tensor_ops::matmul_transposed_gpu(x, gate_, router);
    else tensor_ops::matmul_transposed(x, gate_, router);
    y = Tensor({rows, h});
    if (use_gpu) tensor_ops::zero_gpu(y);
    std::vector<std::vector<std::pair<std::size_t, float>>> assignments(
        apss26::NUM_EXPERTS);
    Tensor one({apss26::NUM_EXPERTS});
    std::vector<std::pair<int, float>> routes;
    const float* router_data = static_cast<const Tensor&>(router).data();
    for (std::size_t t = 0; t < rows; ++t) {
        for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e)
            one[e] = router_data[t * apss26::NUM_EXPERTS + e];
        route(one, routes);
        for (auto [e, w] : routes) assignments[e].emplace_back(t, w);
    }
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
        if (assignments[e].empty()) continue;
        const std::size_t count = assignments[e].size();
        Tensor input({count, h}), output;
        if (use_gpu) {
            std::vector<std::size_t> token_rows(count);
            for (std::size_t i = 0; i < count; ++i)
                token_rows[i] = assignments[e][i].first;
            tensor_ops::gather_rows_gpu(x, token_rows, input);
            experts_[e].forward(input, output, true);
            tensor_ops::scatter_add_rows_gpu(
                output, token_rows, assignments[e][0].second, y);
        } else {
            for (std::size_t i = 0; i < count; ++i)
                for (std::size_t j = 0; j < h; ++j)
                    input.at(i, j) =
                        x.at(assignments[e][i].first, j);
            experts_[e].forward(input, output, false);
            for (std::size_t i = 0; i < count; ++i)
                for (std::size_t j = 0; j < h; ++j)
                    y.at(assignments[e][i].first, j) +=
                        assignments[e][i].second * output.at(i, j);
        }
    }
}

PhiAttention::PhiAttention(const ModelLoader& loader, std::size_t layer_idx)
    : q_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.bias"),
      k_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.bias"),
      v_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.bias"),
      o_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.bias") {}

void PhiAttention::forward(const Tensor& x, Tensor& y, const std::vector<std::size_t>& seq_lens, bool use_gpu) const {
    const std::size_t s = x.size(0);
    Tensor q, k, v;
    q_proj_.forward(x, q, use_gpu);
    k_proj_.forward(x, k, use_gpu);
    v_proj_.forward(x, v, use_gpu);
    Tensor out({s, apss26::NUM_ATTENTION_HEADS * apss26::HEAD_DIM});
    if (use_gpu) {
        tensor_ops::apply_rope_gpu(
            q, k, seq_lens, apss26::NUM_ATTENTION_HEADS,
            apss26::NUM_KV_HEADS, apss26::HEAD_DIM, apss26::ROPE_THETA);
        tensor_ops::attention_gpu(
            q, k, v, out, seq_lens, apss26::NUM_ATTENTION_HEADS,
            apss26::NUM_KV_HEADS, apss26::HEAD_DIM);
        o_proj_.forward(out, y, true);
        return;
    }
    tensor_ops::apply_rope(
        q, k, seq_lens, apss26::NUM_ATTENTION_HEADS,
        apss26::NUM_KV_HEADS, apss26::HEAD_DIM, apss26::ROPE_THETA);
    const std::size_t group =
        apss26::NUM_ATTENTION_HEADS / apss26::NUM_KV_HEADS;

    // Flatten (segment, qi_local) into per-row (segment offset, local position)
    // so the query loop below can be a single parallel loop over global rows.
    std::vector<std::size_t> seg_offset(s), seg_pos(s);
    {
        std::size_t offset = 0;
        for (std::size_t seg_len : seq_lens) {
            for (std::size_t pos = 0; pos < seg_len; ++pos) { seg_offset[offset + pos] = offset; seg_pos[offset + pos] = pos; }
            offset += seg_len;
        }
    }

    // score = q . k is computed once per (qi,ki,head) and reused for the max,
    // softmax-denominator, and weighted-value passes (previously recomputed
    // in each of the three -- see docs/issue.md #2). Also parallelized over
    // (query row, head), since this loop was plain single-threaded CPU code.
#pragma omp parallel for collapse(2)
    for (long long qi_ll = 0; qi_ll < static_cast<long long>(s); ++qi_ll) {
        for (long long qh_ll = 0; qh_ll < static_cast<long long>(apss26::NUM_ATTENTION_HEADS); ++qh_ll) {
            const std::size_t qi = static_cast<std::size_t>(qi_ll);
            const std::size_t qh = static_cast<std::size_t>(qh_ll);
            const std::size_t offset = seg_offset[qi];
            const std::size_t qi_local = seg_pos[qi];
            const std::size_t kh = qh / group;

            // A query attends to at most SLIDING_WINDOW keys, so that is the
            // exact bound on this buffer regardless of sequence length.
            const std::size_t window = std::min<std::size_t>(qi_local + 1, apss26::SLIDING_WINDOW);
            const std::size_t start_local = qi_local - window + 1;
            float scores[apss26::SLIDING_WINDOW];

            float maxv = -std::numeric_limits<float>::infinity();
            for (std::size_t w = 0; w < window; ++w) {
                const std::size_t ki = offset + start_local + w;
                float score = 0.0f;
                for (std::size_t d = 0; d < apss26::HEAD_DIM; ++d) score += q.at(qi, qh * apss26::HEAD_DIM + d) * k.at(ki, kh * apss26::HEAD_DIM + d);
                score /= std::sqrt(static_cast<float>(apss26::HEAD_DIM));
                scores[w] = score;
                maxv = std::max(maxv, score);
            }
            float denom = 0.0f;
            for (std::size_t w = 0; w < window; ++w) { scores[w] = accurate_exp(scores[w] - maxv); denom += scores[w]; }
            for (std::size_t d = 0; d < apss26::HEAD_DIM; ++d) {
                float value = 0.0f;
                for (std::size_t w = 0; w < window; ++w) {
                    const std::size_t ki = offset + start_local + w;
                    value += scores[w] / denom * v.at(ki, kh * apss26::HEAD_DIM + d);
                }
                out.at(qi, qh * apss26::HEAD_DIM + d) = value;
            }
        }
    }
    o_proj_.forward(out, y, use_gpu);
}

PhiDecoderLayer::PhiDecoderLayer(const ModelLoader& loader, std::size_t layer_idx)
    : input_norm_weight_(loader.load("model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight")),
      input_norm_bias_(loader.load("model.layers." + std::to_string(layer_idx) + ".input_layernorm.bias")),
      post_norm_weight_(loader.load("model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight")),
      post_norm_bias_(loader.load("model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.bias")),
      attention_(loader, layer_idx), moe_(loader, layer_idx) {}

void PhiDecoderLayer::forward(const Tensor& x, Tensor& y, const std::vector<std::size_t>& seq_lens, bool use_gpu) const {
    Tensor normed(x.shape()), attn;
    if (use_gpu)
        tensor_ops::layer_norm_gpu(
            x, input_norm_weight_, input_norm_bias_, apss26::NORM_EPS, normed);
    else
        tensor_ops::layer_norm(
            x, input_norm_weight_, input_norm_bias_, apss26::NORM_EPS, normed);
    attention_.forward(normed, attn, seq_lens, use_gpu);
    Tensor post(attn.shape()), ff;
    if (use_gpu) {
        tensor_ops::add_layer_norm_gpu(
            attn, x, post_norm_weight_, post_norm_bias_,
            apss26::NORM_EPS, post);
    } else {
        tensor_ops::add_inplace(attn, x);
        tensor_ops::layer_norm(
            attn, post_norm_weight_, post_norm_bias_, apss26::NORM_EPS, post);
    }
    moe_.forward(post, ff, use_gpu);
    if (use_gpu) tensor_ops::add_inplace_gpu(ff, attn);
    else tensor_ops::add_inplace(ff, attn);
    y = std::move(ff);
}
