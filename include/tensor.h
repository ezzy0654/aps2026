#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape);
    ~Tensor();
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    std::size_t ndim() const { return shape_.size(); }
    std::size_t size() const { return numel_; }
    std::size_t size(std::size_t dim) const;
    const std::vector<std::size_t>& shape() const { return shape_; }

    float* data();
    const float* data() const;
    float& operator[](std::size_t i);
    const float& operator[](std::size_t i) const;

    float& at(std::size_t i);
    float& at(std::size_t i, std::size_t j);
    float& at(std::size_t i, std::size_t j, std::size_t k);
    float& at(std::size_t i, std::size_t j, std::size_t k, std::size_t l);
    const float& at(std::size_t i) const;
    const float& at(std::size_t i, std::size_t j) const;
    const float& at(std::size_t i, std::size_t j, std::size_t k) const;
    const float& at(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const;

    void reshape(std::vector<std::size_t> shape);
    void fill(float value);
    void zero() { fill(0.0f); }
    void prepare_cuda() const;
    void prepare_cuda_bf16_weight(bool with_low_residual) const;
    const float* cuda_data() const;
    float* cuda_data_write();
    const void* cuda_bf16_weight_hi() const { return cuda_bf16_hi_; }
    const void* cuda_bf16_weight_lo() const { return cuda_bf16_lo_; }
    bool has_cuda_bf16_weight() const { return cuda_bf16_hi_ != nullptr; }
    void sync_cuda_to_host() const;

private:
    std::vector<std::size_t> shape_;
    std::size_t numel_ = 0;
    mutable std::vector<float> data_;
    mutable float* cuda_data_ = nullptr;
    mutable void* cuda_bf16_hi_ = nullptr;
    mutable void* cuda_bf16_lo_ = nullptr;
    mutable bool host_valid_ = true;
    mutable bool cuda_valid_ = false;
    std::size_t offset(std::initializer_list<std::size_t> indices) const;
    void ensure_host() const;
    void release_cuda();
};

namespace tensor_ops {
void matmul(const Tensor& a, const Tensor& b, Tensor& c);              // [M,K] x [K,N]
void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c);   // [M,K] x [N,K]
// Shared-memory tiled CUDA matmul, kept separate so the CPU reference path
// (still used by decode) is untouched. Wired into the prefill forward path
// via the use_gpu flag threaded through Linear/PhiMLP/PhiMoE/PhiAttention/
// PhiDecoderLayer/PhiTinyMoEModel::forward (see model.cu generate(), which
// also packs the whole batch into one [total_tokens, HIDDEN] pass -- see
// docs/issue.md #1/#3).
void matmul_transposed_gpu(const Tensor& a, const Tensor& b, Tensor& c);
void matmul_pair_silu_gpu(const Tensor& a, const Tensor& gate_weight,
                          const Tensor& up_weight, Tensor& out);
void matmul_pair_bias_gpu(const Tensor& a,
                          const Tensor& first_weight,
                          const Tensor& first_bias, Tensor& first_out,
                          const Tensor& second_weight,
                          const Tensor& second_bias, Tensor& second_out);
void zero_gpu(Tensor& x);
void add_inplace_gpu(Tensor& a, const Tensor& b);
void add_bias_inplace_gpu(Tensor& a, const Tensor& bias);
void silu_mul_gpu(const Tensor& gate, const Tensor& up, Tensor& out);
void layer_norm_gpu(const Tensor& x, const Tensor& weight, const Tensor& bias,
                    float eps, Tensor& y);
void add_layer_norm_gpu(Tensor& x, const Tensor& residual,
                        const Tensor& weight, const Tensor& bias,
                        float eps, Tensor& y);
void apply_rope_gpu(Tensor& q, Tensor& k,
                    const std::vector<std::size_t>& seq_lens,
                    std::size_t q_heads, std::size_t kv_heads,
                    std::size_t head_dim, float theta);
void attention_gpu(const Tensor& q, const Tensor& k, const Tensor& v,
                   Tensor& out, const std::vector<std::size_t>& seq_lens,
                   std::size_t q_heads, std::size_t kv_heads,
                   std::size_t head_dim);
// Prefix-trie variants of the two ops above. With prefix deduplication the
// rows are trie nodes, so a row's RoPE position is its depth (`row_pos`) and
// the keys it attends to are its ancestor chain, listed in depth order by
// `anc[row * anc_stride + d]`. This removes the expand-to-token-rows /
// contract-back round trip that attention otherwise needs.
void apply_rope_tree_gpu(Tensor& q, Tensor& k,
                         const std::uint32_t* row_pos, std::size_t rows,
                         std::size_t max_seq, std::size_t q_heads,
                         std::size_t kv_heads, std::size_t head_dim,
                         float theta);
void attention_tree_gpu(const Tensor& q, const Tensor& k, const Tensor& v,
                        Tensor& out, const std::uint32_t* row_pos,
                        const std::uint32_t* anc, std::size_t anc_stride,
                        std::size_t rows, std::size_t max_seq,
                        std::size_t q_heads, std::size_t kv_heads,
                        std::size_t head_dim);
void gather_rows_gpu(const Tensor& x, const std::vector<std::size_t>& rows,
                     Tensor& out);

// A row-index map uploaded to the device once and reused across many gathers.
// The std::vector overload above shares one static staging buffer and does a
// blocking H2D copy per call, which is fine for the single end-of-run gather
// but would serialize the pipeline if used per layer.
class RowIndexBuffer {
public:
    RowIndexBuffer() = default;
    explicit RowIndexBuffer(const std::vector<std::size_t>& rows);
    ~RowIndexBuffer();
    RowIndexBuffer(const RowIndexBuffer&) = delete;
    RowIndexBuffer& operator=(const RowIndexBuffer&) = delete;
    RowIndexBuffer(RowIndexBuffer&& other) noexcept;
    RowIndexBuffer& operator=(RowIndexBuffer&& other) noexcept;
    const std::uint32_t* data() const { return ptr_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
private:
    std::uint32_t* ptr_ = nullptr;
    std::size_t size_ = 0;
};
void gather_rows_gpu(const Tensor& x, const RowIndexBuffer& rows, Tensor& out);
void scatter_add_rows_gpu(const Tensor& x,
                          const std::vector<std::size_t>& rows,
                          float scale, Tensor& out);
std::size_t register_moe_weights_gpu(
    const std::vector<const Tensor*>& w1,
    const std::vector<const Tensor*>& w2,
    const std::vector<const Tensor*>& w3);
void moe_forward_grouped_gpu(const Tensor& x, const Tensor& router,
                             std::size_t weights_handle, Tensor& out);
void add_inplace(Tensor& a, const Tensor& b);
void add_bias_inplace(Tensor& a, const Tensor& bias);
void mul(const Tensor& a, const Tensor& b, Tensor& c);
void silu(const Tensor& x, Tensor& y);
void softmax_last_dim(const Tensor& x, Tensor& y);
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                float eps, Tensor& y);
// seq_lens: lengths of the segments packed along q/k's row dimension --
// RoPE position resets to 0 at the start of each segment. A single-sequence
// call passes {rows}.
void apply_rope(Tensor& q, Tensor& k, const std::vector<std::size_t>& seq_lens,
                std::size_t q_heads, std::size_t kv_heads, std::size_t head_dim,
                float theta);
}
