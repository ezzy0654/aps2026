#pragma once

#include <chrono>
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Lightweight per-section timing, enabled only when APS_PROFILE is set in
// the environment. Disabled (the default, and the state during graded
// runs), each ScopedTimer is a single cached-bool check — no clock calls,
// no allocation, no lock. Use APS_PROFILE_SCOPE("name") at the top of a
// block to time it; call profiling::report() once to print the breakdown.
namespace profiling {

bool enabled();
void reset();
void record(const char* name, double seconds);
void report();

class ScopedTimer {
public:
    explicit ScopedTimer(const char* name) : name_(name), active_(enabled()) {
        if (active_) start_ = std::chrono::steady_clock::now();
    }
    ~ScopedTimer() {
        if (active_) {
            // Once real CUDA kernels replace the CPU loops below, this sync
            // makes the section boundary wait for the device work issued in
            // it to actually finish, instead of just timing kernel-launch
            // latency on the host.
            (void)cudaDeviceSynchronize();
            record(name_, std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_).count());
        }
    }
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    const char* name_;
    bool active_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace profiling

#define APS_CONCAT_IMPL(a, b) a##b
#define APS_CONCAT(a, b) APS_CONCAT_IMPL(a, b)
#define APS_PROFILE_SCOPE(name) \
    ::profiling::ScopedTimer APS_CONCAT(_aps_scope_timer_, __LINE__)(name)

namespace detail {
// std::vector<float>'s resize()/assign() value-initialise new elements --
// for float that means writing 0.0f to every element, single-threaded, no
// way to intercept from outside the standard library. This allocator's
// zero-arg construct() does default- instead of value-initialisation
// (a no-op for a scalar like float, leaving it genuinely unformed), so
// resize() on a vector using it becomes allocation-only. Tensor's normal
// constructor re-implements the zero-fill explicitly right after, so every
// existing caller sees identical values and cost; only a caller that
// deliberately skips that explicit fill (see Tensor::uninitialized_parallel)
// gets a different result.
template <typename T>
struct DefaultInitAllocator : std::allocator<T> {
    using Base = std::allocator<T>;
    using Base::Base;
    template <typename U>
    struct rebind { using other = DefaultInitAllocator<U>; };
    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) {
        ::new (static_cast<void*>(p)) U;
    }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        Base::construct(p, std::forward<Args>(args)...);
    }
};
}  // namespace detail

// Tensor is a host-resident value type: the operations below always read
// its inputs from and write its outputs to the host `data_` buffer. A
// Tensor holding model *weights* (loaded once and never mutated) can
// additionally call to_device() so tensor_ops kernels can read it straight
// from device memory instead of re-uploading it on every call — see
// [[project memory]] "weights are preloaded before the timer starts".
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape);
    // Like Tensor(shape), but skips the zero-fill and instead lets an
    // OpenMP-parallel loop fault the pages in (see detail::DefaultInitAllocator
    // above): ~3x faster on this system's page-fault-bound first touch, at
    // the cost of correctness ONLY IF some element is read before being
    // written. Safe exclusively for a buffer every element of which is
    // about to be overwritten regardless -- e.g. a device output landing
    // spot right before the D2H copy that spans it completely. Never use
    // this for a buffer any code might read before writing (accumulators,
    // padding regions, anything relying on Tensor's normal zero-init).
    static Tensor uninitialized_parallel(std::vector<std::size_t> shape);
    // Like uninitialized_parallel, but skips the pre-fault pass too --
    // allocation alone (~0.025 ms even at 131 MB, see uninitialized_parallel's
    // comment) is cheap; the pre-fault is the expensive part. Use this when
    // the caller wants to run that pre-fault on its own schedule (e.g. a
    // background thread overlapped with unrelated GPU work), touching every
    // element via data()/size() before anything reads the buffer.
    static Tensor allocate_uninitialized(std::vector<std::size_t> shape);
    ~Tensor();
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    std::size_t ndim() const { return shape_.size(); }
    std::size_t size() const { return data_.size(); }
    std::size_t size(std::size_t dim) const;
    const std::vector<std::size_t>& shape() const { return shape_; }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    float& operator[](std::size_t i) { return data_[i]; }
    const float& operator[](std::size_t i) const { return data_[i]; }

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

    // Uploads the current host contents to a persistent device buffer
    // (allocated once, re-copied on every call). Intended for weight
    // tensors, called once right after they are loaded.
    void to_device();
    const float* device_data() const { return device_ptr_; }
    bool has_device() const { return device_ptr_ != nullptr; }

private:
    std::vector<std::size_t> shape_;
    std::vector<float, detail::DefaultInitAllocator<float>> data_;
    std::size_t offset(std::initializer_list<std::size_t> indices) const;

    float* device_ptr_ = nullptr;
    std::size_t device_bytes_ = 0;
    void free_device() noexcept;
};

namespace tensor_ops {
void matmul(const Tensor& a, const Tensor& b, Tensor& c);              // [M,K] x [K,N]
void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c);   // [M,K] x [N,K]
void add_inplace(Tensor& a, const Tensor& b);
void add_bias_inplace(Tensor& a, const Tensor& bias);
void mul(const Tensor& a, const Tensor& b, Tensor& c);
void silu(const Tensor& x, Tensor& y);
void softmax_last_dim(const Tensor& x, Tensor& y);
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                float eps, Tensor& y);
// `q`/`k` may hold several sequences concatenated row-wise (a "chunk" —
// see model.cu's forward_chunk); seq_start_of_row[row] gives the row where
// that row's own sequence begins within the chunk, so position is computed
// relative to each sequence rather than the chunk as a whole. For a
// single-sequence call, pass an array of size seq_len filled with 0.
void apply_rope(Tensor& q, Tensor& k, std::size_t seq_len,
                std::size_t q_heads, std::size_t kv_heads, std::size_t head_dim,
                float theta, const int* seq_start_of_row);
// Causal, sliding-window, grouped-query attention: out[qi,qh,:] = softmax
// over ki in (qi-window, qi] of (q.k/sqrt(head_dim)), applied to v.
// seq_start_of_row (see apply_rope above) clamps the window so it never
// looks past the start of qi's own sequence, i.e. never mixes tokens from
// different sequences packed into the same chunk.
void sliding_window_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              std::size_t seq_len, std::size_t q_heads, std::size_t kv_heads,
                              std::size_t head_dim, std::size_t window,
                              const int* seq_start_of_row, Tensor& out);

// Device-resident primitives for PhiMoE's fused expert pipeline: gather ->
// expert FFN (matmul/silu/mul chained on-device) -> scatter-add all run
// back-to-back with no host round trip in between, unlike the Tensor-based
// ops above which upload/download on every single call. Each named Buffer
// is a grow-on-demand device allocation that persists across calls and is
// reused the next time (never freed until process exit); `weight` passed
// to matmul_transposed must already be device-resident (Tensor::to_device()).
namespace device {
enum class Buffer {
    Flat, Y, Router, Input, Output, Gate, Up,
    // Added for the device-resident decoder layer: a layer's hidden state and
    // every intermediate stay in these buffers for all 32 layers, so the only
    // host transfers left in a forward pass are the initial embedding upload
    // and the final logits download.
    Hidden, Normed, Attn, Post, Q, K, V, AttnOut,
    Count
};

void check(cudaError_t err, const char* what);
float* buffer(Buffer which, std::size_t elements);
int* index_buffer(std::size_t elements);
// Distinct from index_buffer, which PhiMoE overwrites once per expert: this
// holds a chunk's token ids for the embedding gather.
int* token_id_buffer(std::size_t elements);
float* weight_buffer(std::size_t elements);
// Per-chunk row metadata. Rows are no longer "sequence i's token j laid out
// contiguously": identical prefixes are computed once and shared, so a row's
// keys are an explicit list rather than the span before it. See model.cu's
// forward_chunk for how the trie is built.
//
//   position[row]                  the row's index within its own sequence,
//                                  which is all RoPE needs
//   path[row * max_len + j]        the row holding the j-th key of `row`,
//                                  for j = 0 .. position[row]
//
// Both are device pointers into persistent buffers, reused across all 32
// decoder layers of one forward_chunk call and overwritten by the next.
struct RowMap {
    const int* position;
    const int* path;
    int max_len;
};
RowMap upload_row_map(const int* position, const int* path,
                      std::size_t rows, std::size_t max_len);

// `bias`, when given, is folded into the GEMM's store: it used to be a
// separate read-modify-write pass over C, moving 765 MB per layer to add an
// 8 KB vector.
void matmul_transposed(const float* d_a, const Tensor& weight, float* d_c, std::size_t m,
                       const Tensor* bias = nullptr);

// One launch covering several independent [rows_e, K] x [N, K] matmuls that
// share K and N -- the 16 MoE experts. Each expert's rows must be contiguous
// in d_a, and its weights must sit at `weight_index * N * K` inside `weights`
// (see PhiMoE's concatenated w1_all_/w3_all_/w2_all_).
//
// Why: an expert alone produces ~220 blocks against the ~490 an RTX 3090
// holds, and the 16 per-expert launches were serialised on the default
// stream, so the GPU ran at under half occupancy 1021 times over. Fusing
// multiplies the block count by 16 without touching the arithmetic -- each
// output element is still accumulated p = 0..K-1 by one thread.
struct GroupTile {
    int row0;           // first row of A/C this block covers
    int row_end;        // one past this expert's last row (bounds the tile)
    int weight_index;   // which expert's weight matrix to read
};
// Host-built tile maps go to a persistent device buffer, overwritten by the
// next call. Callers building two maps (different block_m) should concatenate
// them and offset into the returned pointer.
const GroupTile* upload_group_tiles(const GroupTile* host, std::size_t n);
void matmul_transposed_grouped(const float* d_a, const Tensor& weights,
                               std::size_t experts, float* d_c,
                               const GroupTile* d_tiles, std::size_t num_tiles,
                               std::size_t block_m);
// Plain (uncompensated) accumulation — for the MoE gate only. Its scores feed
// the router's quantized top-2 selection, which agrees with the reference's
// expert choice more often with this than with a more accurate sum; see the
// kernel comment in tensor.cu.
void matmul_transposed_uncompensated(const float* d_a, const Tensor& weight, float* d_c, std::size_t m);
void silu(const float* d_x, float* d_y, std::size_t n);
void mul(const float* d_a, const float* d_b, float* d_c, std::size_t n);
// Fused epilogue for a w1||w3-concatenated grouped GEMM (see PhiMoE in
// layer.h): gateup is [rows, 2*inter], each row's w1 half followed by its
// w3 half. out[r,j] = silu(gateup[r,j]) * gateup[r,inter+j] -- the same
// formula silu()+mul() compute in two passes, done here in one.
void silu_mul_fused(const float* d_gateup, float* d_out, std::size_t rows, std::size_t inter);
void gather_rows(const float* d_src, float* d_dst, const int* d_indices,
                 std::size_t num_rows, std::size_t row_width);
void scatter_add_rows(float* d_dst, const float* d_src, const int* d_indices,
                      const float* d_weights, std::size_t num_rows, std::size_t row_width);
// Direct token -> output scatter for MoE's top-2 combine: host_pos[2*t] and
// host_pos[2*t+1] are the two rows of `d_out` holding token t's two expert
// results (built by the caller while it flattens assignments into the
// expert-major layout `d_out` already has). Replaces a 255 MB zero-fill plus
// 16 serialized per-expert accumulate launches with one pass -- see the
// kernel comment in tensor.cu for why the fixed 0.5/0.5 weights make this
// exactly as order-independent as the memset+accumulate path it replaces.
void scatter_pairs(const float* d_out, const int* host_pos, float* d_y,
                   std::size_t rows, std::size_t row_width);

// Device-pointer counterparts of the Tensor-based ops above. Same kernels,
// minus the per-call upload/download — that transfer, not the arithmetic,
// is what dominates once a whole batch is packed into one chunk.
void add_inplace(float* d_a, const float* d_b, std::size_t n);
void add_bias_inplace(float* d_a, const Tensor& bias, std::size_t n);
void layer_norm(const float* d_x, const Tensor& weight, const Tensor& bias,
                float eps, float* d_y, std::size_t rows, std::size_t h);
// LayerNorm whose first staging pass also applies a residual add, replacing a
// separate add_inplace whose output this kernel then re-read. `d_x` is updated
// in place to hold x + residual, because the next stage needs that sum as its
// own residual. Elementwise order matches add_inplace exactly, so the result
// is bit-identical to running the two separately.
void add_layer_norm(float* d_x, const float* d_residual, const Tensor& weight,
                    const Tensor& bias, float eps, float* d_y,
                    std::size_t rows, std::size_t h);
// rope_table comes from build_rope_table below: max_positions * head_dim/2
// (cos, sin) pairs, indexed by [position * half + j]. The rotation angle
// depends only on those two indices -- not on the data, the head, or the
// layer -- so one table serves every layer of a chunk.
void apply_rope(float* d_q, float* d_k, std::size_t seq_len, std::size_t q_heads,
                std::size_t kv_heads, std::size_t head_dim,
                const RowMap& rows, const float2* rope_table);
// Builds (and returns) that table in a persistent device buffer, overwritten
// by the next call. max_positions must exceed every row's
// `row - seq_start_of_row[row]`. Cheap enough -- 2048 entries for this
// model's inputs -- that forward_chunk rebuilds it per chunk rather than
// caching across calls, which would put a warm-up run's work inside a timed
// one.
const float2* build_rope_table(std::size_t max_positions, std::size_t head_dim, float theta);
void sliding_window_attention(const float* d_q, const float* d_k, const float* d_v,
                              std::size_t seq_len, std::size_t q_heads, std::size_t kv_heads,
                              std::size_t head_dim, std::size_t window,
                              const RowMap& rows, float* d_out);
// Top-2 expert selection for `rows` gate-score vectors that are already on the
// device. Fills host_pairs[2*r] / host_pairs[2*r+1] with row r's first and
// second expert -- the same pair PhiMoE::route returns, bit for bit. Replaces
// a 997 KB blocking download per layer with a 2-int-per-row one.
void route_top2(const float* d_router, int* host_pairs, std::size_t rows);
}  // namespace device
}
