#include "model.h"
#include "config.h"
#include <cstring>
#include <cstdint>
#include <unordered_map>
#include <stdexcept>
#include <vector>

PhiTinyMoEModel::PhiTinyMoEModel(const std::string& model_file)
    : loader_(model_file),
      embeddings_(loader_.load("model.embed_tokens.weight")),
      final_norm_weight_(loader_.load("model.norm.weight")),
      final_norm_bias_(loader_.load("model.norm.bias")),
      lm_head_(loader_, "lm_head.weight", "lm_head.bias") {
    layers_.reserve(apss26::NUM_LAYERS);
    for (std::size_t i = 0; i < apss26::NUM_LAYERS; ++i) layers_.emplace_back(loader_, i);
}

void PhiTinyMoEModel::forward(const std::vector<int>& input_ids, Tensor& logits, bool use_gpu) const {
    if (input_ids.empty()) throw std::invalid_argument("empty input");
    const std::size_t s = input_ids.size();
    if (s > apss26::MAX_POSITION_EMBEDDINGS) throw std::invalid_argument("sequence is too long");
    Tensor hidden({s, apss26::HIDDEN_SIZE});
    for (std::size_t si = 0; si < s; ++si) {
        const int token = input_ids[si];
        if (token < 0 || static_cast<std::size_t>(token) >= apss26::VOCAB_SIZE) throw std::invalid_argument("token out of vocabulary");
        for (std::size_t h = 0; h < apss26::HIDDEN_SIZE; ++h) hidden.at(si, h) = embeddings_.at(static_cast<std::size_t>(token), h);
    }
    const std::vector<std::size_t> seq_lens{s};
    for (const auto& layer : layers_) { Tensor next; layer.forward(hidden, next, seq_lens, use_gpu); hidden = std::move(next); }
    Tensor normed(hidden.shape());
    tensor_ops::layer_norm(hidden, final_norm_weight_, final_norm_bias_, apss26::NORM_EPS, normed);
    Tensor last({1, apss26::HIDDEN_SIZE});
    for (std::size_t h = 0; h < apss26::HIDDEN_SIZE; ++h) last.at(0, h) = normed.at(s - 1, h);
    lm_head_.forward(last, logits, use_gpu);
    logits.reshape({1, apss26::VOCAB_SIZE});
}

// Packs every sequence in the batch into one [total_tokens, HIDDEN] tensor
// (pure prefill, no KV cache -- sequences have no cross-dependency) so every
// Linear/MoE GEMM runs with M = total_tokens instead of M = one sequence's
// length. Attention is the only op that must respect sequence boundaries;
// PhiDecoderLayer::forward takes seq_lens for exactly that. See docs/issue.md #1.
void PhiTinyMoEModel::generate(
    const std::vector<std::vector<int>>& input_ids,
    Tensor& logits) const {
    if (input_ids.empty()) {
        throw std::runtime_error("generate received an empty input batch");
    }

    const std::size_t batch = input_ids.size();
    std::vector<std::size_t> seq_lens(batch), offsets(batch);
    std::size_t total = 0;
    for (std::size_t b = 0; b < batch; ++b) {
        const std::size_t len = input_ids[b].size();
        if (len == 0) throw std::invalid_argument("empty input");
        if (len > apss26::MAX_POSITION_EMBEDDINGS) throw std::invalid_argument("sequence is too long");
        seq_lens[b] = len;
        offsets[b] = total;
        total += len;
    }

    // Prefix deduplication. Attention is causal and the sequences are
    // independent, so a token's hidden state is a pure function of the prefix
    // ending at it: two sequences sharing a prefix produce bit-identical rows
    // there. Collapse the packed token rows onto the nodes of the prefix trie
    // and run every row-wise op (LayerNorm, projections, MoE) on those. For
    // this input that is 19,803 token rows -> 15,583 nodes.
    //
    // `token_node` maps each token row to its node (the expansion attention
    // needs); `node_row` maps each node back to one representative token row.
    // Rows sharing a node are bit-identical, so picking a representative is
    // exact rather than an average.
    std::vector<std::size_t> token_node(total), node_token, node_row;
    node_token.reserve(total);
    node_row.reserve(total);
    {
        std::unordered_map<std::uint64_t, std::uint32_t> children;
        children.reserve(total * 2);
        for (std::size_t b = 0; b < batch; ++b) {
            std::uint32_t parent = 0xFFFFFFFFu;  // virtual root
            for (std::size_t si = 0; si < seq_lens[b]; ++si) {
                const int token = input_ids[b][si];
                if (token < 0 || static_cast<std::size_t>(token) >= apss26::VOCAB_SIZE) throw std::invalid_argument("token out of vocabulary");
                const std::size_t row = offsets[b] + si;
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(parent + 1u) << 32) |
                    static_cast<std::uint32_t>(token);
                auto it = children.find(key);
                std::uint32_t node;
                if (it == children.end()) {
                    node = static_cast<std::uint32_t>(node_token.size());
                    node_token.push_back(static_cast<std::size_t>(token));
                    node_row.push_back(row);
                    children.emplace(key, node);
                } else {
                    node = it->second;
                }
                token_node[row] = node;
                parent = node;
            }
        }
    }
    const std::size_t nodes = node_token.size();

    // Embedding gather runs on the GPU. The embedding table is already
    // device-resident, so gathering there replaces three host-side passes over
    // 255 MB -- allocating and zero-filling the host buffer, the row memcpy,
    // and the H2D upload of the result -- with a 62 KB index upload and one
    // kernel. Those passes were pure host time with the GPU idle, since the
    // hidden states are the first thing every layer needs. The bytes written
    // are the same rows in the same order, so the result is bit-identical.
    Tensor hidden({nodes, apss26::HIDDEN_SIZE});
    tensor_ops::gather_rows_gpu(embeddings_, node_token, hidden);

    const tensor_ops::RowIndexBuffer expand_rows(token_node);
    const tensor_ops::RowIndexBuffer contract_rows(node_row);
    PrefixExpansion expansion;
    expansion.expand = &expand_rows;
    expansion.contract = &contract_rows;
    expansion.num_tokens = total;

    for (const auto& layer : layers_) { Tensor next; layer.forward(hidden, next, seq_lens, /*use_gpu=*/true, expansion); hidden = std::move(next); }

    Tensor normed(hidden.shape());
    tensor_ops::layer_norm_gpu(
        hidden, final_norm_weight_, final_norm_bias_,
        apss26::NORM_EPS, normed);

    Tensor last({batch, apss26::HIDDEN_SIZE});
    std::vector<std::size_t> last_rows(batch);
    for (std::size_t b = 0; b < batch; ++b)
        last_rows[b] = token_node[offsets[b] + seq_lens[b] - 1];
    tensor_ops::gather_rows_gpu(normed, last_rows, last);
    lm_head_.forward(last, logits, /*use_gpu=*/true);
    logits.reshape({batch, apss26::VOCAB_SIZE});
}

void PhiTinyMoEModel::generate_decode(
    const std::vector<std::vector<int>>& input_ids,
    std::size_t max_new_tokens,
    Tensor& logits,
    std::vector<std::vector<int>>& generated_ids) const {
    if (input_ids.empty()) {
        throw std::runtime_error("generate_decode received an empty input batch");
    }
    if (max_new_tokens == 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }

    const std::size_t batch = input_ids.size();
    logits = Tensor({batch, max_new_tokens, apss26::VOCAB_SIZE});
    generated_ids.assign(batch, {});

    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<int> prefix = input_ids[b];
        generated_ids[b].reserve(max_new_tokens);

        for (std::size_t step = 0; step < max_new_tokens; ++step) {
            Tensor one_logits;
            forward(prefix, one_logits);

            int next_token = 0;
            float best_logit = one_logits.at(0, 0);
            for (std::size_t v = 1; v < apss26::VOCAB_SIZE; ++v) {
                const float value = one_logits.at(0, v);
                if (value > best_logit) {
                    best_logit = value;
                    next_token = static_cast<int>(v);
                }
            }

            for (std::size_t v = 0; v < apss26::VOCAB_SIZE; ++v) {
                logits.at(b, step, v) = one_logits.at(0, v);
            }
            generated_ids[b].push_back(next_token);

            if (next_token == apss26::EOS_TOKEN_ID) break;
            prefix.push_back(next_token);
        }
    }
}
