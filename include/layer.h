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
    // additionally combined into w13_all_, [16*896, 4096], one 896-wide
    // grouped launch replacing two (448+448 = 896 = 7*128 exactly, where 448
    // alone pads to 512 in the GEMM's 128-wide tile -- 12.5% wasted
    // columns). Rows are NOT simple [w1; w3] concatenation: they're
    // interleaved in 64-row (w1 chunk, w3 chunk) pairs -- expert e's rows
    // [e*896 + c*128, e*896 + c*128 + 128) hold w1's rows [c*64, c*64+64)
    // then w3's same range, for c = 0..6 -- so the GEMM's SILU_PAIR epilogue
    // (tensor.cu) can read a thread's w1 and w3 outputs for the same
    // interpolation index straight out of its own registers and fold
    // silu(w1)*w3 into the store, rather than materialising [rows, 2*inter]
    // for a separate silu_mul_fused pass to read back.
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
    // Last-layer tail path: only the rows lm_head reads (one per sequence,
    // its terminal trie node) need q/attention/o_proj at all -- nothing
    // downstream reads any other row's output from this layer. k/v still
    // need every row: they're read as ancestor keys by whichever *other*
    // rows attend through this node. d_x_full feeds k_proj/v_proj (rows_full
    // rows, row_map_full for k's RoPE and every row's own key lookup);
    // d_x_tail feeds q_proj (rows_tail rows, row_map_tail -- the same
    // per-node position and ancestor-chain data as row_map_full, just with
    // one row per sequence's terminal node instead of one per trie node).
    // d_y_tail is [rows_tail, hidden].
    void forward_device_tail(const float* d_x_full, const float* d_x_tail, float* d_y_tail,
                             std::size_t rows_full, std::size_t rows_tail,
                             const tensor_ops::device::RowMap& row_map_full,
                             const tensor_ops::device::RowMap& row_map_tail,
                             const float2* rope_table) const;
private:
    Linear q_proj_, k_proj_, v_proj_, o_proj_;
};

class PhiDecoderLayer {
public:
    PhiDecoderLayer(const ModelLoader& loader, std::size_t layer_idx);
    // Reads [rows, HIDDEN_SIZE] from d_x, writes the layer output to d_y.
    // d_y must not alias d_x.
    //
    // The layer no longer closes its own second residual. It returns the raw
    // MoE output in d_y and leaves its attention residual in Buffer::Attn;
    // the caller hands that back as `d_carry` on the next call, where the
    // input LayerNorm's staging pass absorbs the add (see add_layer_norm).
    // d_carry is null for the first layer, and the last layer's pair is
    // closed by the final LayerNorm in model.cu. d_x is updated in place to
    // x + carry, which is what the post-attention residual then reads.
    void forward_device(float* d_x, const float* d_carry, float* d_y, std::size_t rows,
                        const tensor_ops::device::RowMap& row_map,
                        const float2* rope_table) const;
    // Only the last layer calls this. d_x is the FULL [rows_full, hidden]
    // hidden state, mutated in place to x+carry exactly like forward_device
    // (input_norm still runs on every row -- k_proj/v_proj need it). d_y_tail
    // is [rows_tail, hidden]: the raw MoE output for just the rows lm_head
    // reads. This layer's own attention residual (attn + x_tail) for those
    // same rows -- what forward_device would leave as the carry -- ends up
    // in Buffer::AttnTail; model.cu re-fetches that pointer after this call
    // returns, the same way it re-fetches Buffer::Attn after forward_device.
    void forward_device_tail(float* d_x, const float* d_carry, float* d_y_tail,
                             std::size_t rows_full, std::size_t rows_tail,
                             const tensor_ops::device::RowMap& row_map_full,
                             const tensor_ops::device::RowMap& row_map_tail,
                             const float2* rope_table, const int* d_tail_index) const;
private:
    Tensor input_norm_weight_, input_norm_bias_;
    Tensor post_norm_weight_, post_norm_bias_;
    PhiAttention attention_;
    PhiMoE moe_;
};
