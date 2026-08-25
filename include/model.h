#pragma once

#include "layer.h"
#include <string>
#include <vector>

class PhiTinyMoEModel {
public:
    explicit PhiTinyMoEModel(const std::string& model_file);

    void forward(const std::vector<int>& input_ids, Tensor& logits, bool use_gpu = false) const;

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
};
