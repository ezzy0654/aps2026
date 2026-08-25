#pragma once

#include "config.h"
#include "model_loader.h"
#include "tensor.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

class Linear {
public:
    Linear() = default;
    Linear(const ModelLoader& loader, const std::string& weight, const std::string& bias = "");
    void forward(const Tensor& x, Tensor& y, bool use_gpu = false) const;
private:
    Tensor weight_;
    Tensor bias_;
};

class PhiMLP {
public:
    PhiMLP(const ModelLoader& loader, const std::string& prefix);
    void forward(const Tensor& x, Tensor& y, bool use_gpu = false) const;
private:
    Tensor w1_, w2_, w3_;
};

class PhiMoE {
public:
    PhiMoE(const ModelLoader& loader, std::size_t layer_idx);
    void forward(const Tensor& x, Tensor& y, bool use_gpu = false) const;
private:
    Tensor gate_;
    std::vector<PhiMLP> experts_;
    void route(const Tensor& logits, std::vector<std::pair<int, float>>& routes) const;
};

class PhiAttention {
public:
    PhiAttention(const ModelLoader& loader, std::size_t layer_idx);
    // seq_lens: lengths of the sequences packed along x's row dimension;
    // attention (and RoPE) never crosses a segment boundary. Single-sequence
    // callers pass {x.size(0)}.
    void forward(const Tensor& x, Tensor& y, const std::vector<std::size_t>& seq_lens, bool use_gpu = false) const;
private:
    Linear q_proj_, k_proj_, v_proj_, o_proj_;
};

class PhiDecoderLayer {
public:
    PhiDecoderLayer(const ModelLoader& loader, std::size_t layer_idx);
    void forward(const Tensor& x, Tensor& y, const std::vector<std::size_t>& seq_lens, bool use_gpu = false) const;
private:
    Tensor input_norm_weight_, input_norm_bias_;
    Tensor post_norm_weight_, post_norm_bias_;
    PhiAttention attention_;
    PhiMoE moe_;
};
