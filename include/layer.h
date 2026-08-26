#pragma once

#include "config.h"
#include "model_loader.h"
#include "tensor.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// Everything below the model level now works on raw device pointers rather
// than Tensors. A decoder layer's hidden state is uploaded once per chunk
// and stays on the GPU across all 32 layers; only the initial embedding and
// the final logits cross the PCIe bus. (Transfers, not arithmetic, dominated
// once a whole batch became one chunk.) Buffers come from
// tensor_ops::device::buffer(), which hands out grow-on-demand allocations
// that persist between calls.

class Linear {
public:
    Linear() = default;
    Linear(const ModelLoader& loader, const std::string& weight, const std::string& bias = "");
    void forward(const Tensor& x, Tensor& y) const;
    // Device-resident form: reads `rows` x in_features from d_x, writes
    // `rows` x out_features to d_y (caller sizes both).
    void forward_device(const float* d_x, float* d_y, std::size_t rows) const;
    std::size_t out_features() const { return weight_.size(0); }
private:
    Tensor weight_;
    Tensor bias_;
};

class PhiMLP {
public:
    PhiMLP(const ModelLoader& loader, const std::string& prefix);
    // The whole w1/silu/w3/mul/w2 chain runs back-to-back on the GPU with no
    // host round trip, so this can be called directly from PhiMoE's fused
    // expert pipeline.
    void forward_device(const float* d_x, float* d_y, std::size_t rows) const;
private:
    Tensor w1_, w2_, w3_;
};

class PhiMoE {
public:
    PhiMoE(const ModelLoader& loader, std::size_t layer_idx);
    // d_x and d_y are [rows, hidden] device buffers; d_y must not alias d_x.
    void forward_device(const float* d_x, float* d_y, std::size_t rows, std::size_t hidden) const;
private:
    Tensor gate_;
    // All 16 experts' weights concatenated along their row axis, so the
    // expert FFNs run as one grouped launch each instead of 16 serialised
    // ones: w2_all_ is [16*4096, 448], expert e occupying rows
    // [e*4096, (e+1)*4096). Concatenation happens at construction, before
    // the timer starts -- a weight-layout change, which is the allowed
    // kind. See tensor.h's matmul_transposed_grouped.
    //
    // w1 and w3 (each expert's gate/up projection, both [448, 4096]) are
    // additionally concatenated ALONG N into w13_all_, [16*896, 4096] --
    // expert e's rows [e*896, e*896+448) are w1, [e*896+448, (e+1)*896) are
    // w3. 448+448 = 896 = 7*128 exactly, where 448 alone pads to 512 in the
    // GEMM's 128-wide tile (12.5% wasted columns); one 896-wide grouped
    // launch also replaces two, and its epilogue feeds directly into
    // device::silu_mul_fused instead of separate silu/mul kernels.
    Tensor w13_all_, w2_all_;
    // Reads this token's NUM_EXPERTS gate scores, writes its top-2 picks.
    void route(const float* logits, std::pair<int, float> routes[2]) const;
};

class PhiAttention {
public:
    PhiAttention(const ModelLoader& loader, std::size_t layer_idx);
    // row_map carries this chunk's per-row position and key-path arrays -- see
    // tensor.h's RowMap. Rows are prefix-trie nodes, so a row's keys are its
    // ancestor chain rather than the span before it. rope_table is the chunk's
    // shared (cos, sin) rotation table from build_rope_table; one table serves
    // every layer, since the angle depends only on position and pair index.
    void forward_device(const float* d_x, float* d_y, std::size_t rows,
                        const tensor_ops::device::RowMap& row_map,
                        const float2* rope_table) const;
private:
    Linear q_proj_, k_proj_, v_proj_, o_proj_;
};

class PhiDecoderLayer {
public:
    PhiDecoderLayer(const ModelLoader& loader, std::size_t layer_idx);
    // Reads [rows, HIDDEN_SIZE] from d_x, writes the layer output to d_y.
    // d_y must not alias d_x.
    void forward_device(const float* d_x, float* d_y, std::size_t rows,
                        const tensor_ops::device::RowMap& row_map,
                        const float2* rope_table) const;
private:
    Tensor input_norm_weight_, input_norm_bias_;
    Tensor post_norm_weight_, post_norm_bias_;
    PhiAttention attention_;
    PhiMoE moe_;
};
