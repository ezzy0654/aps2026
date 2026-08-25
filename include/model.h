#pragma once

#include "layer.h"
#include <string>
#include <vector>

class PhiTinyMoEModel {
public:
    explicit PhiTinyMoEModel(const std::string& model_file);

    void forward(const std::vector<int>& input_ids, Tensor& logits) const;

    void generate(const std::vector<std::vector<int>>& input_ids,
                  Tensor& logits) const;

    void generate_decode(const std::vector<std::vector<int>>& input_ids,
                         std::size_t max_new_tokens,
                         Tensor& logits,
                         std::vector<std::vector<int>>& generated_ids) const;

private:
    ModelLoader loader_;
    Tensor embeddings_;
    Tensor final_norm_weight_, final_norm_bias_;
    Linear lm_head_;
    std::vector<PhiDecoderLayer> layers_;

    // Runs the full decoder stack over several sequences packed row-wise
    // into one chunk (bigger M for every Linear/MoE matmul than running
    // sequences one at a time). Attention/RoPE stay sequence-aware via an
    // internally built seq_start_of_row array, so sequences never bleed into
    // each other.
    //
    // Each sequence's final-token logits are written straight into
    // out[dest_rows[i]] -- one device-to-host copy per row, landing in the
    // caller's final layout. Going through an intermediate [chunk, VOCAB]
    // Tensor and re-scattering it cost a 131MB zero-fill plus ~33M
    // bounds-checked element copies.
    void forward_chunk(const std::vector<std::vector<int>>& chunk_ids,
                       const std::vector<std::size_t>& dest_rows, Tensor& out) const;
};
