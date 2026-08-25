#include "tensor.h"
#include "config.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <string>
#include <unordered_map>
namespace {
void tensor_cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}
}

Tensor::Tensor(std::vector<std::size_t> shape) : shape_(std::move(shape)) {
    numel_ = 1;
    for (std::size_t d : shape_) numel_ *= d;
}

Tensor::~Tensor() { release_cuda(); }

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), numel_(other.numel_) {
    other.ensure_host();
    data_ = other.data_;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    other.ensure_host();
    release_cuda();
    shape_ = other.shape_;
    numel_ = other.numel_;
    data_ = other.data_;
    host_valid_ = true;
    cuda_valid_ = false;
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), numel_(other.numel_),
      data_(std::move(other.data_)), cuda_data_(other.cuda_data_),
      cuda_bf16_hi_(other.cuda_bf16_hi_), cuda_bf16_lo_(other.cuda_bf16_lo_),
      host_valid_(other.host_valid_),
      cuda_valid_(other.cuda_valid_) {
    other.cuda_data_ = nullptr;
    other.cuda_bf16_hi_ = nullptr;
    other.cuda_bf16_lo_ = nullptr;
    other.numel_ = 0;
    other.host_valid_ = true;
    other.cuda_valid_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    release_cuda();
    shape_ = std::move(other.shape_);
    numel_ = other.numel_;
    data_ = std::move(other.data_);
    cuda_data_ = other.cuda_data_;
    cuda_bf16_hi_ = other.cuda_bf16_hi_;
    cuda_bf16_lo_ = other.cuda_bf16_lo_;
    host_valid_ = other.host_valid_;
    cuda_valid_ = other.cuda_valid_;
    other.cuda_data_ = nullptr;
    other.cuda_bf16_hi_ = nullptr;
    other.cuda_bf16_lo_ = nullptr;
    other.numel_ = 0;
    other.host_valid_ = true;
    other.cuda_valid_ = false;
    return *this;
}

void Tensor::release_cuda() {
    if (cuda_data_) cudaFreeAsync(cuda_data_, 0);
    if (cuda_bf16_hi_) cudaFreeAsync(cuda_bf16_hi_, 0);
    if (cuda_bf16_lo_) cudaFreeAsync(cuda_bf16_lo_, 0);
    cuda_data_ = nullptr;
    cuda_bf16_hi_ = nullptr;
    cuda_bf16_lo_ = nullptr;
    cuda_valid_ = false;
}

void Tensor::ensure_host() const {
    if (data_.size() != numel_) data_.assign(numel_, 0.0f);
    if (host_valid_) return;
    if (!cuda_data_ || !cuda_valid_)
        throw std::runtime_error("tensor has no valid storage");
    tensor_cuda_check(cudaMemcpy(data_.data(), cuda_data_,
                                 numel_ * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy tensor D2H");
    host_valid_ = true;
}

void Tensor::prepare_cuda() const {
    if (numel_ == 0) return;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        cudaGetLastError();
        return;
    }
    if (!cuda_data_)
        tensor_cuda_check(cudaMallocAsync(&cuda_data_,
                                           numel_ * sizeof(float), 0),
                          "cudaMallocAsync tensor");
    if (!cuda_valid_) {
        ensure_host();
        tensor_cuda_check(cudaMemcpy(cuda_data_, data_.data(),
                                     numel_ * sizeof(float),
                                     cudaMemcpyHostToDevice),
                          "cudaMemcpy tensor H2D");
        cuda_valid_ = true;
    }
}

namespace {
__global__ void k_pack_bf16_weight(
    const float* src, __nv_bfloat16* hi, __nv_bfloat16* lo,
    std::size_t n, std::size_t k, std::size_t k_tiles) {
    const std::size_t packed_idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = ((n + 15) / 16) * k_tiles * 256;
    if (packed_idx >= total) return;
    const std::size_t tile_elem = packed_idx & 255;
    const std::size_t tile = packed_idx >> 8;
    const std::size_t kt = tile % k_tiles;
    const std::size_t nt = tile / k_tiles;
    const std::size_t local_n = tile_elem / 16;
    const std::size_t local_k = tile_elem % 16;
    const std::size_t row = nt * 16 + local_n;
    const std::size_t col = kt * 16 + local_k;
    const float value = row < n && col < k ? src[row * k + col] : 0.0f;
    const __nv_bfloat16 upper = __float2bfloat16_rn(value);
    hi[packed_idx] = upper;
    if (lo)
        lo[packed_idx] = __float2bfloat16_rn(
            value - __bfloat162float(upper));
}
}

void Tensor::prepare_cuda_bf16_weight(bool with_low_residual) const {
    if (shape_.size() != 2)
        throw std::invalid_argument("BF16 packing requires a matrix");
    if (cuda_bf16_hi_ && (!with_low_residual || cuda_bf16_lo_)) return;
    prepare_cuda();
    const std::size_t n_tiles = (shape_[0] + 15) / 16;
    const std::size_t k_tiles = (shape_[1] + 15) / 16;
    const std::size_t elems = n_tiles * k_tiles * 256;
    if (!cuda_bf16_hi_)
        tensor_cuda_check(cudaMallocAsync(&cuda_bf16_hi_,
                                           elems * sizeof(__nv_bfloat16), 0),
                          "cudaMallocAsync BF16 weight");
    if (with_low_residual && !cuda_bf16_lo_)
        tensor_cuda_check(cudaMallocAsync(&cuda_bf16_lo_,
                                           elems * sizeof(__nv_bfloat16), 0),
                          "cudaMallocAsync BF16 residual");
    k_pack_bf16_weight<<<(elems + 255) / 256, 256>>>(
        cuda_data_, static_cast<__nv_bfloat16*>(cuda_bf16_hi_),
        static_cast<__nv_bfloat16*>(cuda_bf16_lo_),
        shape_[0], shape_[1], k_tiles);
    tensor_cuda_check(cudaGetLastError(), "k_pack_bf16_weight launch");
    // Packing is ordered on the same stream, so the immutable FP32 device
    // copy can be released without synchronizing. Host FP32 remains intact
    // for the CPU/decode path.
    tensor_cuda_check(cudaFreeAsync(cuda_data_, 0),
                      "cudaFreeAsync packed FP32 weight");
    cuda_data_ = nullptr;
    cuda_valid_ = false;
}

const float* Tensor::cuda_data() const {
    prepare_cuda();
    return cuda_data_;
}

float* Tensor::cuda_data_write() {
    if (!cuda_data_)
        tensor_cuda_check(cudaMallocAsync(&cuda_data_,
                                           numel_ * sizeof(float), 0),
                          "cudaMallocAsync tensor");
    cuda_valid_ = true;
    host_valid_ = false;
    return cuda_data_;
}

void Tensor::sync_cuda_to_host() const { ensure_host(); }

float* Tensor::data() {
    ensure_host();
    cuda_valid_ = false;
    return data_.data();
}

const float* Tensor::data() const {
    ensure_host();
    return data_.data();
}

float& Tensor::operator[](std::size_t i) {
    ensure_host();
    cuda_valid_ = false;
    return data_[i];
}

const float& Tensor::operator[](std::size_t i) const {
    ensure_host();
    return data_[i];
}

std::size_t Tensor::size(std::size_t dim) const {
    if (dim >= shape_.size()) throw std::out_of_range("Tensor dimension");
    return shape_[dim];
}

std::size_t Tensor::offset(std::initializer_list<std::size_t> indices) const {
    if (indices.size() != shape_.size()) throw std::invalid_argument("Tensor rank mismatch");
    std::size_t out = 0, stride = 1;
    auto it = indices.end();
    for (std::size_t d = shape_.size(); d-- > 0;) {
        --it;
        if (*it >= shape_[d]) throw std::out_of_range("Tensor index");
        out += *it * stride;
        stride *= shape_[d];
    }
    return out;
}

float& Tensor::at(std::size_t i) {
    ensure_host(); cuda_valid_ = false; return data_[offset({i})];
}
float& Tensor::at(std::size_t i, std::size_t j) {
    ensure_host(); cuda_valid_ = false; return data_[offset({i, j})];
}
float& Tensor::at(std::size_t i, std::size_t j, std::size_t k) {
    ensure_host(); cuda_valid_ = false; return data_[offset({i, j, k})];
}
float& Tensor::at(std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    ensure_host(); cuda_valid_ = false; return data_[offset({i, j, k, l})];
}
const float& Tensor::at(std::size_t i) const {
    ensure_host(); return data_[offset({i})];
}
const float& Tensor::at(std::size_t i, std::size_t j) const {
    ensure_host(); return data_[offset({i, j})];
}
const float& Tensor::at(std::size_t i, std::size_t j, std::size_t k) const {
    ensure_host(); return data_[offset({i, j, k})];
}
const float& Tensor::at(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const {
    ensure_host(); return data_[offset({i, j, k, l})];
}

void Tensor::reshape(std::vector<std::size_t> shape) {
    std::size_t n = 1;
    for (std::size_t d : shape) n *= d;
    if (n != numel_) throw std::invalid_argument("reshape changes tensor size");
    shape_ = std::move(shape);
}

void Tensor::fill(float value) {
    if (data_.size() != numel_) data_.resize(numel_);
    std::fill(data_.begin(), data_.end(), value);
    host_valid_ = true;
    cuda_valid_ = false;
}

namespace tensor_ops {

void zero_gpu(Tensor& x) {
    tensor_cuda_check(cudaMemsetAsync(x.cuda_data_write(), 0,
                                      x.size() * sizeof(float), 0),
                      "cudaMemsetAsync tensor");
}
// Match the reference CUDA/PyTorch path: evaluate exp in double precision
// and round only the final result back to float.
static inline float accurate_exp(float x) {
    return static_cast<float>(std::exp(static_cast<double>(x)));
}

void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c) {
    const std::size_t m = a.size(0), k = a.size(1), n = b.size(0);
    if (b.size(1) != k || c.size(0) != m || c.size(1) != n) throw std::invalid_argument("matmul_transposed shape");
    c.zero();
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(m); ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t p = 0; p < k; ++p) sum += a.at(i, p) * b.at(j, p);
            c.at(i, j) = sum;
        }
    }
}

void add_inplace(Tensor& a, const Tensor& b) {
    if (a.size() != b.size()) throw std::invalid_argument("add shape");
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(a.size()); ++i) a[i] += b[i];
}

void add_bias_inplace(Tensor& a, const Tensor& bias) {
    if (bias.ndim() != 1 || a.size(a.ndim() - 1) != bias.size(0)) throw std::invalid_argument("bias shape");
    const std::size_t h = bias.size(0);
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(a.size()); ++i) a[i] += bias[i % h];
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.size() != b.size() || c.size() != a.size()) throw std::invalid_argument("mul shape");
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(a.size()); ++i) c[i] = a[i] * b[i];
}

void silu(const Tensor& x, Tensor& y) {
    if (x.size() != y.size()) throw std::invalid_argument("silu shape");
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(x.size()); ++i) y[i] = x[i] / (1.0f + accurate_exp(-x[i]));
}

void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& y) {
    const std::size_t h = x.size(x.ndim() - 1), rows = x.size() / h;
    if (weight.size() != h || bias.size() != h || y.size() != x.size()) throw std::invalid_argument("layer norm shape");
#pragma omp parallel for
    for (long long r = 0; r < static_cast<long long>(rows); ++r) {
        const std::size_t base = static_cast<std::size_t>(r) * h;
        float mean = 0.0f;
        for (std::size_t j = 0; j < h; ++j) mean += x[base + j];
        mean /= static_cast<float>(h);
        float var = 0.0f;
        for (std::size_t j = 0; j < h; ++j) { const float d = x[base + j] - mean; var += d * d; }
        const float inv = 1.0f / std::sqrt(var / static_cast<float>(h) + eps);
        for (std::size_t j = 0; j < h; ++j) y[base + j] = (x[base + j] - mean) * inv * weight[j] + bias[j];
    }
}

void apply_rope(Tensor& q, Tensor& k, const std::vector<std::size_t>& seq_lens,
                std::size_t q_heads, std::size_t kv_heads, std::size_t head_dim, float theta) {
    const std::size_t half = head_dim / 2;
    std::size_t offset = 0;
    for (std::size_t seg_len : seq_lens) {
        for (std::size_t pos = 0; pos < seg_len; ++pos) {
            const std::size_t s = offset + pos;
            for (std::size_t h = 0; h < q_heads; ++h) for (std::size_t j = 0; j < half; ++j) {
                const float inv = std::pow(theta, -2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
                const float c = std::cos(static_cast<float>(pos) * inv), sn = std::sin(static_cast<float>(pos) * inv);
                const float x0 = q.at(s, h * head_dim + j), x1 = q.at(s, h * head_dim + j + half);
                q.at(s, h * head_dim + j) = x0 * c - x1 * sn;
                q.at(s, h * head_dim + j + half) = x1 * c + x0 * sn;
            }
            for (std::size_t h = 0; h < kv_heads; ++h) for (std::size_t j = 0; j < half; ++j) {
                const float inv = std::pow(theta, -2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
                const float c = std::cos(static_cast<float>(pos) * inv), sn = std::sin(static_cast<float>(pos) * inv);
                const float x0 = k.at(s, h * head_dim + j), x1 = k.at(s, h * head_dim + j + half);
                k.at(s, h * head_dim + j) = x0 * c - x1 * sn;
                k.at(s, h * head_dim + j + half) = x1 * c + x0 * sn;
            }
        }
        offset += seg_len;
    }
}

namespace {
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Growable device scratch buffer, reused across calls instead of
// cudaMalloc/cudaFree every matmul_transposed_gpu invocation.
struct DeviceScratch {
    float* ptr = nullptr;
    std::size_t capacity = 0;
};

float* ensure_scratch(DeviceScratch& buf, std::size_t elems) {
    if (buf.capacity < elems) {
        if (buf.ptr) cudaFree(buf.ptr);
        buf.ptr = nullptr;
        check_cuda(cudaMalloc(&buf.ptr, elems * sizeof(float)), "cudaMalloc(scratch)");
        buf.capacity = elems;
    }
    return buf.ptr;
}

// Weight tensors (Linear/PhiMLP/PhiMoE gate weights) are loaded once at
// model construction and never mutated during generate(), so their host
// pointer is a stable cache key -- upload each weight matrix to the GPU at
// most once instead of on every forward call.
//
// The upload is lazy (first use is inside generate()), so its cost is paid
// inside the measured region, not hoisted out of it.
//
// The full weight set is ~14GB against 24GB of device memory, so it fits --
// but a batch that touches every expert leaves little headroom. Returns
// nullptr instead of throwing when the cache allocation fails, so the caller
// can stage that weight through scratch memory and the run still completes.
float* cached_device_weight(const Tensor& b) {
    struct Entry { float* ptr; std::size_t elems; };
    static std::unordered_map<const float*, Entry> cache;
    const float* key = b.data();
    auto it = cache.find(key);
    if (it != cache.end() && it->second.elems == b.size()) return it->second.ptr;
    float* dptr = nullptr;
    if (cudaMalloc(&dptr, b.size() * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();  // clear the sticky-free OOM status
        return nullptr;
    }
    check_cuda(cudaMemcpy(dptr, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy(weight cache H2D)");
    cache[key] = Entry{dptr, b.size()};
    return dptr;
}

// c[row,col] = sum_p a[row,p] * b[col,p], i.e. C = A * B^T, A:[M,K] B:[N,K]
// row-major (K contiguous). Shared-memory tiled so global loads of A and B
// are coalesced (K is the fast dimension for both) and each loaded tile is
// reused TILE times instead of re-streamed per output thread -- see
// docs/issue.md #3. TILE=16 matches the MoE router's N=NUM_EXPERTS=16 with
// zero tile waste and keeps blocks small for occupancy on the many
// small-M per-expert GEMMs. (An earlier attempt measured this slower, but
// that was with the unfixed attention loop -- see docs/issue.md #2 -- eating
// the whole budget; re-measured clean it's a net win.)
constexpr int TILE = 16;

__global__ void k_matmul_transposed_tiled(const float* __restrict__ a, const float* __restrict__ b,
                                           float* __restrict__ c,
                                           std::size_t m, std::size_t k, std::size_t n) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const std::size_t row = blockIdx.y * TILE + threadIdx.y;
    const std::size_t col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    const std::size_t k_tiles = (k + TILE - 1) / TILE;
    for (std::size_t t = 0; t < k_tiles; ++t) {
        const std::size_t a_col = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;

        const std::size_t b_row = blockIdx.x * TILE + threadIdx.y;
        const std::size_t b_col = t * TILE + threadIdx.x;
        Bs[threadIdx.y][threadIdx.x] = (b_row < n && b_col < k) ? b[b_row * k + b_col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int p = 0; p < TILE; ++p) sum += As[threadIdx.y][p] * Bs[threadIdx.x][p];

        __syncthreads();
    }

    if (row < m && col < n) c[row * n + col] = sum;
}
constexpr int BIG_BM = 64;
constexpr int BIG_BN = 64;
constexpr int BIG_BK = 32;
constexpr int BIG_BX = 32;
constexpr int BIG_BY = 8;
constexpr int BIG_TM = BIG_BM / BIG_BY;
constexpr int BIG_TN = BIG_BN / BIG_BX;

// sm_86 splits a 128 KB unified L1/shared block per SM in fixed carveout
// steps. A kernel holding 32 KB of static shared gets a 32 KB carveout by
// default, which caps residency at one block per SM no matter how small the
// register footprint is. Ask for half the block (64 KB) so two of them fit;
// the register budget set by __launch_bounds__ is what holds it at two.
void prefer_shared_carveout(const void* kernel) {
    // cudaSharedmemCarveoutMaxShared asks for the whole 100 KB shared
    // partition. A 50% request was silently kept at the default carveout,
    // so the shared limiter stayed at one block per SM.
    if (cudaFuncSetAttribute(
            kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared) != cudaSuccess)
        cudaGetLastError();
}

__device__ __forceinline__ void cp_async_16(
    void* dst, const void* src, int valid_bytes) {
#if __CUDA_ARCH__ >= 800
    const unsigned smem = static_cast<unsigned>(
        __cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
        "r"(smem), "l"(src), "r"(valid_bytes));
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_one() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

#ifdef USE_TC
__global__ void k_matmul_transposed_bf16_wmma(
    const float* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_hi,
    const __nv_bfloat16* __restrict__ b_lo,
    float* __restrict__ c,
    std::size_t m, std::size_t k, std::size_t n) {
    using namespace nvcuda;
    __shared__ __align__(16) __nv_bfloat16 a_hi[64 * 16];
    __shared__ __align__(16) __nv_bfloat16 a_lo[64 * 16];
    __shared__ __align__(16) float c_tile[64 * 64];
    const int warp = threadIdx.x / 32;
    const int warp_m = warp / 2;
    const int warp_n0 = (warp % 2) * 2;
    const std::size_t row0 = static_cast<std::size_t>(blockIdx.y) * 64;
    const std::size_t col0 = static_cast<std::size_t>(blockIdx.x) * 64;
    const std::size_t k_tiles = (k + 15) / 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);
    for (std::size_t kt = 0; kt < k_tiles; ++kt) {
        for (int idx = threadIdx.x; idx < 64 * 16; idx += blockDim.x) {
            const std::size_t lr = static_cast<std::size_t>(idx) / 16;
            const std::size_t lk = static_cast<std::size_t>(idx) % 16;
            const std::size_t row = row0 + lr;
            const std::size_t p = kt * 16 + lk;
            const float value = row < m && p < k ? a[row * k + p] : 0.0f;
            const __nv_bfloat16 upper = __float2bfloat16_rn(value);
            a_hi[idx] = upper;
            a_lo[idx] = __float2bfloat16_rn(
                value - __bfloat162float(upper));
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       __nv_bfloat16, wmma::row_major> af_hi;
        wmma::load_matrix_sync(af_hi, a_hi + warp_m * 256, 16);
        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       __nv_bfloat16, wmma::row_major> af_lo;
        wmma::load_matrix_sync(af_lo, a_lo + warp_m * 256, 16);
        for (int nn = 0; nn < 2; ++nn) {
            const std::size_t n_tile =
                static_cast<std::size_t>(blockIdx.x) * 4 + warp_n0 + nn;
            const std::size_t packed_tile = (n_tile * k_tiles + kt) * 256;
            wmma::fragment<wmma::matrix_b, 16, 16, 16,
                           __nv_bfloat16, wmma::col_major> bf_hi;
            wmma::load_matrix_sync(bf_hi, b_hi + packed_tile, 16);
            wmma::mma_sync(acc[nn], af_hi, bf_hi, acc[nn]);
            wmma::fragment<wmma::matrix_b, 16, 16, 16,
                           __nv_bfloat16, wmma::col_major> bf_lo;
            wmma::load_matrix_sync(bf_lo, b_lo + packed_tile, 16);
            wmma::mma_sync(acc[nn], af_hi, bf_lo, acc[nn]);
            wmma::mma_sync(acc[nn], af_lo, bf_hi, acc[nn]);
#ifdef USE_TC_ALL
            wmma::mma_sync(acc[nn], af_lo, bf_lo, acc[nn]);
#endif
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(
        c_tile + warp_m * 16 * 64 + warp_n0 * 16,
        acc[0], 64, wmma::mem_row_major);
    wmma::store_matrix_sync(
        c_tile + warp_m * 16 * 64 + (warp_n0 + 1) * 16,
        acc[1], 64, wmma::mem_row_major);
    __syncthreads();
    for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
        const std::size_t row = row0 + static_cast<std::size_t>(idx) / 64;
        const std::size_t col = col0 + static_cast<std::size_t>(idx) % 64;
        if (row < m && col < n) c[row * n + col] = c_tile[idx];
    }
}
#endif

// Wider N tile: 64 x 128 output per block instead of 64 x 64, so each thread
// owns BIG_TM x WIDE_TN = 8 x 4 = 32 accumulators instead of 16. Per K step a
// thread reads 8 A values and 4 B values for 32 FMAs (0.375 shared loads per
// FMA) where the 64-wide kernel reads 10 for 16 (0.625). Same two-stage
// cp.async pipeline, same 4-float XOR swizzle, and the same K order, so the
// results match k_matmul_transposed_register_blocked_async exactly.
constexpr int WIDE_BN = 128;
constexpr int WIDE_TN = WIDE_BN / BIG_BX;

__global__ void __launch_bounds__(BIG_BX * BIG_BY, 2)
k_matmul_transposed_wide_async(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c,
    std::size_t m, std::size_t k, std::size_t n) {
    __shared__ __align__(16) float As[2][BIG_BM][BIG_BK];
    __shared__ __align__(16) float Bs[2][WIDE_BN][BIG_BK];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BIG_BX + tx;
    const std::size_t block_row =
        static_cast<std::size_t>(blockIdx.y) * BIG_BM;
    const std::size_t block_col =
        static_cast<std::size_t>(blockIdx.x) * WIDE_BN;
    const std::size_t row0 = block_row + ty * BIG_TM;
    const std::size_t col0 = block_col + tx * WIDE_TN;
    float acc[BIG_TM][WIDE_TN] = {};
    const std::size_t tiles = k / BIG_BK;

    auto issue = [&](int stage, std::size_t p0) {
        constexpr int a_vectors = BIG_BM * BIG_BK / 4;
        for (int vec = tid; vec < a_vectors; vec += BIG_BX * BIG_BY) {
            const int lr = vec / (BIG_BK / 4);
            const int chunk = vec % (BIG_BK / 4);
            const std::size_t row = block_row + lr;
            const float* src = row < m
                ? a + row * k + p0 + chunk * 4 : a;
            cp_async_16(&As[stage][lr][chunk * 4], src,
                        row < m ? 16 : 0);
        }
        constexpr int b_vectors = WIDE_BN * BIG_BK / 4;
        for (int vec = tid; vec < b_vectors; vec += BIG_BX * BIG_BY) {
            const int lc = vec / (BIG_BK / 4);
            const int chunk = vec % (BIG_BK / 4);
            const int swizzled = chunk ^ ((lc >> 1) & 7);
            const std::size_t col = block_col + lc;
            const float* src = col < n
                ? b + col * k + p0 + chunk * 4 : b;
            cp_async_16(&Bs[stage][lc][swizzled * 4], src,
                        col < n ? 16 : 0);
        }
        cp_async_commit();
    };

    issue(0, 0);
    for (std::size_t tile = 0; tile < tiles; ++tile) {
        const int stage = tile & 1;
        if (tile + 1 < tiles) {
            issue(stage ^ 1, (tile + 1) * BIG_BK);
            cp_async_wait_one();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BIG_BK; ++p) {
            float bv[WIDE_TN];
#pragma unroll
            for (int cc = 0; cc < WIDE_TN; ++cc) {
                const int lc = tx * WIDE_TN + cc;
                const int physical_p =
                    ((p / 4) ^ ((lc >> 1) & 7)) * 4 + (p & 3);
                bv[cc] = Bs[stage][lc][physical_p];
            }
#pragma unroll
            for (int r = 0; r < BIG_TM; ++r) {
                const float av = As[stage][ty * BIG_TM + r][p];
#pragma unroll
                for (int cc = 0; cc < WIDE_TN; ++cc)
                    acc[r][cc] += av * bv[cc];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < BIG_TM; ++r)
#pragma unroll
        for (int cc = 0; cc < WIDE_TN; ++cc) {
            const std::size_t row = row0 + r, col = col0 + cc;
            if (row < m && col < n) c[row * n + col] = acc[r][cc];
        }
}

__global__ void __launch_bounds__(BIG_BX * BIG_BY, 2)
k_matmul_transposed_register_blocked_async(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c,
    std::size_t m, std::size_t k, std::size_t n) {
    __shared__ __align__(16) float As[2][BIG_BM][BIG_BK];
    __shared__ __align__(16) float Bs[2][BIG_BN][BIG_BK];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BIG_BX + tx;
    const std::size_t block_row =
        static_cast<std::size_t>(blockIdx.y) * BIG_BM;
    const std::size_t block_col =
        static_cast<std::size_t>(blockIdx.x) * BIG_BN;
    const std::size_t row0 = block_row + ty * BIG_TM;
    const std::size_t col0 = block_col + tx * BIG_TN;
    float acc[BIG_TM][BIG_TN] = {};
    const std::size_t tiles = k / BIG_BK;

    auto issue = [&](int stage, std::size_t p0) {
        constexpr int vectors_per_tile = BIG_BM * BIG_BK / 4;
        for (int vec = tid; vec < vectors_per_tile;
             vec += BIG_BX * BIG_BY) {
            const int lr = vec / (BIG_BK / 4);
            const int chunk = vec % (BIG_BK / 4);
            const std::size_t row = block_row + lr;
            const float* src = row < m
                ? a + row * k + p0 + chunk * 4 : a;
            cp_async_16(&As[stage][lr][chunk * 4], src,
                        row < m ? 16 : 0);
        }
        for (int vec = tid; vec < vectors_per_tile;
             vec += BIG_BX * BIG_BY) {
            const int lc = vec / (BIG_BK / 4);
            const int chunk = vec % (BIG_BK / 4);
            const int swizzled = chunk ^ ((lc >> 1) & 7);
            const std::size_t col = block_col + lc;
            const float* src = col < n
                ? b + col * k + p0 + chunk * 4 : b;
            cp_async_16(&Bs[stage][lc][swizzled * 4], src,
                        col < n ? 16 : 0);
        }
        cp_async_commit();
    };

    issue(0, 0);
    for (std::size_t tile = 0; tile < tiles; ++tile) {
        const int stage = tile & 1;
        if (tile + 1 < tiles) {
            issue(stage ^ 1, (tile + 1) * BIG_BK);
            cp_async_wait_one();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BIG_BK; ++p) {
#pragma unroll
            for (int r = 0; r < BIG_TM; ++r) {
                const float av = As[stage][ty * BIG_TM + r][p];
#pragma unroll
                for (int cc = 0; cc < BIG_TN; ++cc) {
                    const int lc = tx * BIG_TN + cc;
                    const int physical_p =
                        ((p / 4) ^ ((lc >> 1) & 7)) * 4 + (p & 3);
                    acc[r][cc] += av * Bs[stage][lc][physical_p];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < BIG_TM; ++r)
#pragma unroll
        for (int cc = 0; cc < BIG_TN; ++cc) {
            const std::size_t row = row0 + r, col = col0 + cc;
            if (row < m && col < n) c[row * n + col] = acc[r][cc];
        }
}

__global__ void k_matmul_transposed_register_blocked(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c,
    std::size_t m, std::size_t k, std::size_t n) {
    __shared__ float As[BIG_BM][BIG_BK + 1];
    __shared__ float Bs[BIG_BN][BIG_BK + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BIG_BX + tx;
    const std::size_t row0 =
        static_cast<std::size_t>(blockIdx.y) * BIG_BM + ty * BIG_TM;
    const std::size_t col0 =
        static_cast<std::size_t>(blockIdx.x) * BIG_BN + tx * BIG_TN;
    float acc[BIG_TM][BIG_TN] = {};

    for (std::size_t p0 = 0; p0 < k; p0 += BIG_BK) {
        for (int idx = tid; idx < BIG_BM * BIG_BK; idx += BIG_BX * BIG_BY) {
            const int lr = idx / BIG_BK;
            const int lp = idx % BIG_BK;
            const std::size_t row =
                static_cast<std::size_t>(blockIdx.y) * BIG_BM + lr;
            const std::size_t p = p0 + lp;
            As[lr][lp] = row < m && p < k ? a[row * k + p] : 0.0f;
        }
        for (int idx = tid; idx < BIG_BN * BIG_BK; idx += BIG_BX * BIG_BY) {
            const int lc = idx / BIG_BK;
            const int lp = idx % BIG_BK;
            const std::size_t col =
                static_cast<std::size_t>(blockIdx.x) * BIG_BN + lc;
            const std::size_t p = p0 + lp;
            Bs[lc][lp] = col < n && p < k ? b[col * k + p] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BIG_BK; ++p) {
#pragma unroll
            for (int r = 0; r < BIG_TM; ++r) {
                const float av = As[ty * BIG_TM + r][p];
#pragma unroll
                for (int cc = 0; cc < BIG_TN; ++cc)
                    acc[r][cc] += av * Bs[tx * BIG_TN + cc][p];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < BIG_TM; ++r)
#pragma unroll
        for (int cc = 0; cc < BIG_TN; ++cc) {
            const std::size_t row = row0 + r;
            const std::size_t col = col0 + cc;
            if (row < m && col < n) c[row * n + col] = acc[r][cc];
        }
}

__global__ void k_matmul_pair_silu_register_blocked(
    const float* __restrict__ a,
    const float* __restrict__ gate_weight,
    const float* __restrict__ up_weight,
    float* __restrict__ out,
    std::size_t m, std::size_t k, std::size_t n) {
    __shared__ float As[BIG_BM][BIG_BK + 1];
    __shared__ float Bs[BIG_BN][BIG_BK + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BIG_BX + tx;
    const std::size_t row0 =
        static_cast<std::size_t>(blockIdx.y) * BIG_BM + ty * BIG_TM;
    const std::size_t col0 =
        static_cast<std::size_t>(blockIdx.x) * BIG_BN + tx * BIG_TN;
    float gate_acc[BIG_TM][BIG_TN] = {};
    float up_acc[BIG_TM][BIG_TN] = {};

    for (std::size_t p0 = 0; p0 < k; p0 += BIG_BK) {
        for (int idx = tid; idx < BIG_BM * BIG_BK;
             idx += BIG_BX * BIG_BY) {
            const int lr = idx / BIG_BK;
            const int lp = idx % BIG_BK;
            const std::size_t row =
                static_cast<std::size_t>(blockIdx.y) * BIG_BM + lr;
            const std::size_t p = p0 + lp;
            As[lr][lp] = row < m && p < k ? a[row * k + p] : 0.0f;
        }
        for (int idx = tid; idx < BIG_BN * BIG_BK;
             idx += BIG_BX * BIG_BY) {
            const int lc = idx / BIG_BK;
            const int lp = idx % BIG_BK;
            const std::size_t col =
                static_cast<std::size_t>(blockIdx.x) * BIG_BN + lc;
            const std::size_t p = p0 + lp;
            Bs[lc][lp] = col < n && p < k
                ? gate_weight[col * k + p] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BIG_BK; ++p) {
#pragma unroll
            for (int r = 0; r < BIG_TM; ++r) {
                const float av = As[ty * BIG_TM + r][p];
#pragma unroll
                for (int cc = 0; cc < BIG_TN; ++cc)
                    gate_acc[r][cc] +=
                        av * Bs[tx * BIG_TN + cc][p];
            }
        }
        __syncthreads();

        for (int idx = tid; idx < BIG_BN * BIG_BK;
             idx += BIG_BX * BIG_BY) {
            const int lc = idx / BIG_BK;
            const int lp = idx % BIG_BK;
            const std::size_t col =
                static_cast<std::size_t>(blockIdx.x) * BIG_BN + lc;
            const std::size_t p = p0 + lp;
            Bs[lc][lp] = col < n && p < k
                ? up_weight[col * k + p] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BIG_BK; ++p) {
#pragma unroll
            for (int r = 0; r < BIG_TM; ++r) {
                const float av = As[ty * BIG_TM + r][p];
#pragma unroll
                for (int cc = 0; cc < BIG_TN; ++cc)
                    up_acc[r][cc] += av * Bs[tx * BIG_TN + cc][p];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < BIG_TM; ++r)
#pragma unroll
        for (int cc = 0; cc < BIG_TN; ++cc) {
            const std::size_t row = row0 + r;
            const std::size_t col = col0 + cc;
            if (row < m && col < n) {
                const float x = gate_acc[r][cc];
                const float activated = x /
                    (1.0f + static_cast<float>(
                        exp(static_cast<double>(-x))));
                out[row * n + col] = activated * up_acc[r][cc];
            }
        }
}

__global__ void k_matmul_pair_bias_register_blocked(
    const float* __restrict__ a,
    const float* __restrict__ first_weight,
    const float* __restrict__ first_bias,
    float* __restrict__ first_out,
    const float* __restrict__ second_weight,
    const float* __restrict__ second_bias,
    float* __restrict__ second_out,
    std::size_t m, std::size_t k, std::size_t n) {
    // One shared tile per weight, staged before a single barrier: two
    // __syncthreads() per K tile instead of four, and one shared read of each
    // A value instead of one per weight.
    __shared__ float As[BIG_BM][BIG_BK + 1];
    __shared__ float Bs[BIG_BN][BIG_BK + 1];
    __shared__ float Bs2[BIG_BN][BIG_BK + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BIG_BX + tx;
    const std::size_t row0 =
        static_cast<std::size_t>(blockIdx.y) * BIG_BM + ty * BIG_TM;
    const std::size_t col0 =
        static_cast<std::size_t>(blockIdx.x) * BIG_BN + tx * BIG_TN;
    float first_acc[BIG_TM][BIG_TN] = {};
    float second_acc[BIG_TM][BIG_TN] = {};

    for (std::size_t p0 = 0; p0 < k; p0 += BIG_BK) {
        for (int idx = tid; idx < BIG_BM * BIG_BK;
             idx += BIG_BX * BIG_BY) {
            const int lr = idx / BIG_BK;
            const int lp = idx % BIG_BK;
            const std::size_t row =
                static_cast<std::size_t>(blockIdx.y) * BIG_BM + lr;
            const std::size_t p = p0 + lp;
            As[lr][lp] = row < m && p < k ? a[row * k + p] : 0.0f;
        }
        for (int idx = tid; idx < BIG_BN * BIG_BK;
             idx += BIG_BX * BIG_BY) {
            const int lc = idx / BIG_BK;
            const int lp = idx % BIG_BK;
            const std::size_t col =
                static_cast<std::size_t>(blockIdx.x) * BIG_BN + lc;
            const std::size_t p = p0 + lp;
            Bs[lc][lp] = col < n && p < k
                ? first_weight[col * k + p] : 0.0f;
            Bs2[lc][lp] = col < n && p < k
                ? second_weight[col * k + p] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BIG_BK; ++p)
#pragma unroll
            for (int r = 0; r < BIG_TM; ++r) {
                const float av = As[ty * BIG_TM + r][p];
#pragma unroll
                for (int cc = 0; cc < BIG_TN; ++cc) {
                    const int lc = tx * BIG_TN + cc;
                    first_acc[r][cc] += av * Bs[lc][p];
                    second_acc[r][cc] += av * Bs2[lc][p];
                }
            }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < BIG_TM; ++r)
#pragma unroll
        for (int cc = 0; cc < BIG_TN; ++cc) {
            const std::size_t row = row0 + r;
            const std::size_t col = col0 + cc;
            if (row < m && col < n) {
                first_out[row * n + col] =
                    first_acc[r][cc] + first_bias[col];
                second_out[row * n + col] =
                    second_acc[r][cc] + second_bias[col];
            }
        }
}

__global__ void __launch_bounds__(BIG_BX * BIG_BY, 2)
k_matmul_pair_bias_wide(
    const float* __restrict__ a,
    const float* __restrict__ first_weight,
    const float* __restrict__ first_bias,
    float* __restrict__ first_out,
    const float* __restrict__ second_weight,
    const float* __restrict__ second_bias,
    float* __restrict__ second_out,
    std::size_t m, std::size_t k, std::size_t n) {
    // One shared tile per weight, staged before a single barrier: two
    // __syncthreads() per K tile instead of four, and one shared read of each
    // A value instead of one per weight.
    __shared__ float As[BIG_BM][BIG_BK + 1];
    __shared__ float Bs[WIDE_BN][BIG_BK + 1];
    __shared__ float Bs2[WIDE_BN][BIG_BK + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BIG_BX + tx;
    const std::size_t row0 =
        static_cast<std::size_t>(blockIdx.y) * BIG_BM + ty * BIG_TM;
    const std::size_t col0 =
        static_cast<std::size_t>(blockIdx.x) * WIDE_BN + tx * WIDE_TN;
    float first_acc[BIG_TM][WIDE_TN] = {};
    float second_acc[BIG_TM][WIDE_TN] = {};

    for (std::size_t p0 = 0; p0 < k; p0 += BIG_BK) {
        for (int idx = tid; idx < BIG_BM * BIG_BK;
             idx += BIG_BX * BIG_BY) {
            const int lr = idx / BIG_BK;
            const int lp = idx % BIG_BK;
            const std::size_t row =
                static_cast<std::size_t>(blockIdx.y) * BIG_BM + lr;
            const std::size_t p = p0 + lp;
            As[lr][lp] = row < m && p < k ? a[row * k + p] : 0.0f;
        }
        for (int idx = tid; idx < WIDE_BN * BIG_BK;
             idx += BIG_BX * BIG_BY) {
            const int lc = idx / BIG_BK;
            const int lp = idx % BIG_BK;
            const std::size_t col =
                static_cast<std::size_t>(blockIdx.x) * WIDE_BN + lc;
            const std::size_t p = p0 + lp;
            Bs[lc][lp] = col < n && p < k
                ? first_weight[col * k + p] : 0.0f;
            Bs2[lc][lp] = col < n && p < k
                ? second_weight[col * k + p] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BIG_BK; ++p)
#pragma unroll
            for (int r = 0; r < BIG_TM; ++r) {
                const float av = As[ty * BIG_TM + r][p];
#pragma unroll
                for (int cc = 0; cc < WIDE_TN; ++cc) {
                    const int lc = tx * WIDE_TN + cc;
                    first_acc[r][cc] += av * Bs[lc][p];
                    second_acc[r][cc] += av * Bs2[lc][p];
                }
            }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < BIG_TM; ++r)
#pragma unroll
        for (int cc = 0; cc < WIDE_TN; ++cc) {
            const std::size_t row = row0 + r;
            const std::size_t col = col0 + cc;
            if (row < m && col < n) {
                first_out[row * n + col] =
                    first_acc[r][cc] + first_bias[col];
                second_out[row * n + col] =
                    second_acc[r][cc] + second_bias[col];
            }
        }
}

struct MoeWeightSet {
    const float** w1 = nullptr;
    const float** w2 = nullptr;
    const float** w3 = nullptr;
};

std::vector<MoeWeightSet>& moe_weight_registry() {
    static std::vector<MoeWeightSet> registry;
    return registry;
}

struct MoeScratch {
    std::uint8_t* routes = nullptr;
    std::uint32_t* counts = nullptr;
    std::uint32_t* offsets = nullptr;
    std::uint32_t* assignment_rows = nullptr;
    std::uint32_t* route_positions = nullptr;
    std::uint32_t* work_expert = nullptr;
    std::uint32_t* work_start = nullptr;
    std::size_t row_capacity = 0;
    std::size_t work_capacity = 0;
};

MoeScratch& moe_scratch(std::size_t rows) {
    static MoeScratch scratch;
    const std::size_t max_work = (2 * rows + BIG_BM - 1) / BIG_BM +
                                 apss26::NUM_EXPERTS;
    if (scratch.row_capacity >= rows && scratch.work_capacity >= max_work)
        return scratch;
    if (scratch.routes) cudaFree(scratch.routes);
    if (scratch.counts) cudaFree(scratch.counts);
    if (scratch.offsets) cudaFree(scratch.offsets);
    if (scratch.assignment_rows) cudaFree(scratch.assignment_rows);
    if (scratch.route_positions) cudaFree(scratch.route_positions);
    if (scratch.work_expert) cudaFree(scratch.work_expert);
    if (scratch.work_start) cudaFree(scratch.work_start);
    check_cuda(cudaMalloc(&scratch.routes, 2 * rows * sizeof(std::uint8_t)),
               "cudaMalloc MoE routes");
    check_cuda(cudaMalloc(&scratch.counts,
                          apss26::NUM_EXPERTS * sizeof(std::uint32_t)),
               "cudaMalloc MoE counts");
    check_cuda(cudaMalloc(&scratch.offsets,
                          (apss26::NUM_EXPERTS + 1) *
                              sizeof(std::uint32_t)),
               "cudaMalloc MoE offsets");
    check_cuda(cudaMalloc(&scratch.assignment_rows,
                          2 * rows * sizeof(std::uint32_t)),
               "cudaMalloc MoE assignment rows");
    check_cuda(cudaMalloc(&scratch.route_positions,
                          2 * rows * sizeof(std::uint32_t)),
               "cudaMalloc MoE route positions");
    check_cuda(cudaMalloc(&scratch.work_expert,
                          max_work * sizeof(std::uint32_t)),
               "cudaMalloc MoE work experts");
    check_cuda(cudaMalloc(&scratch.work_start,
                          max_work * sizeof(std::uint32_t)),
               "cudaMalloc MoE work starts");
    scratch.row_capacity = rows;
    scratch.work_capacity = max_work;
    return scratch;
}

__global__ void k_moe_top2(const float* router, std::uint8_t* routes,
                           std::uint32_t* counts, std::size_t rows) {
    const std::size_t t =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= rows) return;
    float scores[apss26::NUM_EXPERTS];
#pragma unroll
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
        const float score = router[t * apss26::NUM_EXPERTS + e];
        const float rounded = floorf(fabsf(score) /
            apss26::ROUTER_SCORE_QUANTUM + 0.5f) *
            apss26::ROUTER_SCORE_QUANTUM;
        scores[e] = score < 0.0f ? -rounded : rounded;
    }
    int first = -1;
    float first_value = -3.402823466e+38F;
#pragma unroll
    for (int e = 0; e < static_cast<int>(apss26::NUM_EXPERTS); ++e) {
        if (first < 0 || scores[e] > first_value + apss26::ROUTER_TIE_EPS) {
            first = e;
            first_value = scores[e];
        }
    }
    int second = -1;
    float second_value = -3.402823466e+38F;
#pragma unroll
    for (int e = 0; e < static_cast<int>(apss26::NUM_EXPERTS); ++e) {
        if (e == first) continue;
        if (second < 0 ||
            scores[e] > second_value + apss26::ROUTER_TIE_EPS) {
            second = e;
            second_value = scores[e];
        }
    }
    routes[2 * t] = static_cast<std::uint8_t>(first);
    routes[2 * t + 1] = static_cast<std::uint8_t>(second);
    atomicAdd(counts + first, 1u);
    atomicAdd(counts + second, 1u);
}

__global__ void k_moe_offsets_work(
    const std::uint32_t* counts, std::uint32_t* offsets,
    std::uint32_t* work_expert, std::uint32_t* work_start) {
    __shared__ std::uint32_t work_offsets[apss26::NUM_EXPERTS + 1];
    if (threadIdx.x == 0) {
        offsets[0] = 0;
        work_offsets[0] = 0;
        for (std::size_t expert = 0; expert < apss26::NUM_EXPERTS;
             ++expert) {
            offsets[expert + 1] = offsets[expert] + counts[expert];
            work_offsets[expert + 1] = work_offsets[expert] +
                (counts[expert] + BIG_BM - 1) / BIG_BM;
        }
    }
    __syncthreads();
    const unsigned e = threadIdx.x;
    if (e < apss26::NUM_EXPERTS) {
        const std::uint32_t chunks =
            (counts[e] + BIG_BM - 1) / BIG_BM;
        for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
            const std::uint32_t wi = work_offsets[e] + chunk;
            work_expert[wi] = e;
            work_start[wi] = offsets[e] + chunk * BIG_BM;
        }
    }
}

__global__ void k_moe_fill_stable(
    const std::uint8_t* routes, const std::uint32_t* offsets,
    std::uint32_t* assignment_rows, std::uint32_t* route_positions,
    std::size_t rows) {
    const unsigned expert = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31;
    const unsigned warp = tid >> 5;
    __shared__ std::uint32_t base;
    __shared__ std::uint32_t warp_counts[8];
    __shared__ std::uint32_t warp_offsets[9];
    if (tid == 0) base = offsets[expert];
    __syncthreads();
    for (std::size_t tile = 0; tile < rows; tile += blockDim.x) {
        const std::size_t t = tile + tid;
        int slot = -1;
        if (t < rows) {
            if (routes[2 * t] == expert) slot = 0;
            else if (routes[2 * t + 1] == expert) slot = 1;
        }
        const unsigned mask = __ballot_sync(0xffffffffu, slot >= 0);
        if (lane == 0) warp_counts[warp] = __popc(mask);
        __syncthreads();
        if (tid == 0) {
            warp_offsets[0] = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w)
                warp_offsets[w + 1] = warp_offsets[w] + warp_counts[w];
        }
        __syncthreads();
        if (slot >= 0) {
            const unsigned lower =
                lane == 0 ? 0u : ((1u << lane) - 1u);
            const std::uint32_t position = base + warp_offsets[warp] +
                                           __popc(mask & lower);
            assignment_rows[position] = static_cast<std::uint32_t>(t);
            route_positions[2 * t + static_cast<unsigned>(slot)] = position;
        }
        __syncthreads();
        if (tid == 0) base += warp_offsets[8];
        __syncthreads();
    }
}

__global__ void k_moe_gather_compact(
    const float* x, const std::uint32_t* assignment_rows, float* out,
    std::size_t assignments, std::size_t h) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < assignments * h)
        out[i] = x[static_cast<std::size_t>(assignment_rows[i / h]) * h +
                   i % h];
}

__global__ void k_moe_pair_silu_grouped(
    const float* a, const float* const* gate_weights,
    const float* const* up_weights, float* out,
    const std::uint32_t* offsets, const std::uint32_t* work_expert,
    const std::uint32_t* work_start, std::size_t max_work,
    std::size_t k, std::size_t n) {
    const std::size_t wi = blockIdx.y;
    if (wi >= max_work) return;
    const std::uint32_t expert = work_expert[wi];
    if (expert >= apss26::NUM_EXPERTS) return;
    const std::size_t start = work_start[wi];
    const std::size_t end = offsets[expert + 1];
    // One shared tile per weight. Staging both before a single barrier
    // halves the __syncthreads() count per K tile (4 -> 2) and lets each A
    // value be read from shared once instead of once per weight.
    __shared__ float As[BIG_BM][BIG_BK + 1];
    __shared__ float Bs[BIG_BN][BIG_BK + 1];
    __shared__ float Bs2[BIG_BN][BIG_BK + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BIG_BX + tx;
    const std::size_t row0 = start + ty * BIG_TM;
    const std::size_t col0 =
        static_cast<std::size_t>(blockIdx.x) * BIG_BN + tx * BIG_TN;
    float gate_acc[BIG_TM][BIG_TN] = {};
    float up_acc[BIG_TM][BIG_TN] = {};
    const float* gate_weight = gate_weights[expert];
    const float* up_weight = up_weights[expert];
    for (std::size_t p0 = 0; p0 < k; p0 += BIG_BK) {
        for (int idx = tid; idx < BIG_BM * BIG_BK;
             idx += BIG_BX * BIG_BY) {
            const int lr = idx / BIG_BK, lp = idx % BIG_BK;
            const std::size_t row = start + lr, p = p0 + lp;
            As[lr][lp] = row < end && p < k ? a[row * k + p] : 0.0f;
        }
        for (int idx = tid; idx < BIG_BN * BIG_BK;
             idx += BIG_BX * BIG_BY) {
            const int lc = idx / BIG_BK, lp = idx % BIG_BK;
            const std::size_t col =
                static_cast<std::size_t>(blockIdx.x) * BIG_BN + lc;
            const std::size_t p = p0 + lp;
            Bs[lc][lp] = col < n && p < k
                ? gate_weight[col * k + p] : 0.0f;
            Bs2[lc][lp] = col < n && p < k
                ? up_weight[col * k + p] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BIG_BK; ++p)
#pragma unroll
            for (int r = 0; r < BIG_TM; ++r) {
                const float av = As[ty * BIG_TM + r][p];
#pragma unroll
                for (int cc = 0; cc < BIG_TN; ++cc) {
                    const int lc = tx * BIG_TN + cc;
                    gate_acc[r][cc] += av * Bs[lc][p];
                    up_acc[r][cc] += av * Bs2[lc][p];
                }
            }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < BIG_TM; ++r)
#pragma unroll
        for (int cc = 0; cc < BIG_TN; ++cc) {
            const std::size_t row = row0 + r, col = col0 + cc;
            if (row < end && col < n) {
                const float x = gate_acc[r][cc];
                out[row * n + col] =
                    x / (1.0f + static_cast<float>(
                        exp(static_cast<double>(-x)))) * up_acc[r][cc];
            }
        }
}

__global__ void k_moe_matmul_grouped(
    const float* a, const float* const* weights, float* out,
    const std::uint32_t* offsets, const std::uint32_t* work_expert,
    const std::uint32_t* work_start, std::size_t max_work,
    std::size_t k, std::size_t n) {
    const std::size_t wi = blockIdx.y;
    if (wi >= max_work) return;
    const std::uint32_t expert = work_expert[wi];
    if (expert >= apss26::NUM_EXPERTS) return;
    const std::size_t start = work_start[wi];
    const std::size_t end = offsets[expert + 1];
    __shared__ float As[BIG_BM][BIG_BK + 1];
    __shared__ float Bs[BIG_BN][BIG_BK + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BIG_BX + tx;
    const std::size_t row0 = start + ty * BIG_TM;
    const std::size_t col0 =
        static_cast<std::size_t>(blockIdx.x) * BIG_BN + tx * BIG_TN;
    float acc[BIG_TM][BIG_TN] = {};
    const float* weight = weights[expert];
    for (std::size_t p0 = 0; p0 < k; p0 += BIG_BK) {
        for (int idx = tid; idx < BIG_BM * BIG_BK;
             idx += BIG_BX * BIG_BY) {
            const int lr = idx / BIG_BK, lp = idx % BIG_BK;
            const std::size_t row = start + lr, p = p0 + lp;
            As[lr][lp] = row < end && p < k ? a[row * k + p] : 0.0f;
        }
        for (int idx = tid; idx < BIG_BN * BIG_BK;
             idx += BIG_BX * BIG_BY) {
            const int lc = idx / BIG_BK, lp = idx % BIG_BK;
            const std::size_t col =
                static_cast<std::size_t>(blockIdx.x) * BIG_BN + lc;
            const std::size_t p = p0 + lp;
            Bs[lc][lp] = col < n && p < k ? weight[col * k + p] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BIG_BK; ++p)
#pragma unroll
            for (int r = 0; r < BIG_TM; ++r) {
                const float av = As[ty * BIG_TM + r][p];
#pragma unroll
                for (int cc = 0; cc < BIG_TN; ++cc)
                    acc[r][cc] += av * Bs[tx * BIG_TN + cc][p];
            }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < BIG_TM; ++r)
#pragma unroll
        for (int cc = 0; cc < BIG_TN; ++cc) {
            const std::size_t row = row0 + r, col = col0 + cc;
            if (row < end && col < n) out[row * n + col] = acc[r][cc];
        }
}

// Same grouped W2 GEMM, but with the two-stage cp.async pipeline the Q/O
// projection already uses: the next K tile is copied global->shared while the
// current one is being multiplied. Requires k % BIG_BK == 0 so every tile is
// full and every 16-byte copy stays aligned. The K accumulation order is
// unchanged -- p0 ascending, then p = 0..BIG_BK-1 -- so results are identical
// to k_moe_matmul_grouped.
__global__ void __launch_bounds__(BIG_BX * BIG_BY, 2)
k_moe_matmul_grouped_async(
    const float* a, const float* const* weights, float* out,
    const std::uint32_t* offsets, const std::uint32_t* work_expert,
    const std::uint32_t* work_start, std::size_t max_work,
    std::size_t k, std::size_t n) {
    const std::size_t wi = blockIdx.y;
    if (wi >= max_work) return;
    const std::uint32_t expert = work_expert[wi];
    if (expert >= apss26::NUM_EXPERTS) return;
    const std::size_t start = work_start[wi];
    const std::size_t end = offsets[expert + 1];
    __shared__ __align__(16) float As[2][BIG_BM][BIG_BK];
    __shared__ __align__(16) float Bs[2][BIG_BN][BIG_BK];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BIG_BX + tx;
    const std::size_t block_col =
        static_cast<std::size_t>(blockIdx.x) * BIG_BN;
    const std::size_t row0 = start + ty * BIG_TM;
    const std::size_t col0 = block_col + tx * BIG_TN;
    float acc[BIG_TM][BIG_TN] = {};
    const float* weight = weights[expert];
    const std::size_t tiles = k / BIG_BK;

    auto issue = [&](int stage, std::size_t p0) {
        constexpr int vectors_per_tile = BIG_BM * BIG_BK / 4;
        for (int vec = tid; vec < vectors_per_tile;
             vec += BIG_BX * BIG_BY) {
            const int lr = vec / (BIG_BK / 4);
            const int chunk = vec % (BIG_BK / 4);
            const std::size_t row = start + lr;
            const float* src = row < end
                ? a + row * k + p0 + chunk * 4 : a;
            cp_async_16(&As[stage][lr][chunk * 4], src,
                        row < end ? 16 : 0);
        }
        for (int vec = tid; vec < vectors_per_tile;
             vec += BIG_BX * BIG_BY) {
            const int lc = vec / (BIG_BK / 4);
            const int chunk = vec % (BIG_BK / 4);
            const int swizzled = chunk ^ ((lc >> 1) & 7);
            const std::size_t col = block_col + lc;
            const float* src = col < n
                ? weight + col * k + p0 + chunk * 4 : weight;
            cp_async_16(&Bs[stage][lc][swizzled * 4], src,
                        col < n ? 16 : 0);
        }
        cp_async_commit();
    };

    issue(0, 0);
    for (std::size_t tile = 0; tile < tiles; ++tile) {
        const int stage = tile & 1;
        if (tile + 1 < tiles) {
            issue(stage ^ 1, (tile + 1) * BIG_BK);
            cp_async_wait_one();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BIG_BK; ++p) {
#pragma unroll
            for (int r = 0; r < BIG_TM; ++r) {
                const float av = As[stage][ty * BIG_TM + r][p];
#pragma unroll
                for (int cc = 0; cc < BIG_TN; ++cc) {
                    const int lc = tx * BIG_TN + cc;
                    const int physical_p =
                        ((p / 4) ^ ((lc >> 1) & 7)) * 4 + (p & 3);
                    acc[r][cc] += av * Bs[stage][lc][physical_p];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < BIG_TM; ++r)
#pragma unroll
        for (int cc = 0; cc < BIG_TN; ++cc) {
            const std::size_t row = row0 + r, col = col0 + cc;
            if (row < end && col < n) out[row * n + col] = acc[r][cc];
        }
}

__global__ void __launch_bounds__(BIG_BX * BIG_BY, 2)
k_moe_matmul_grouped_wide_async(
    const float* a, const float* const* weights, float* out,
    const std::uint32_t* offsets, const std::uint32_t* work_expert,
    const std::uint32_t* work_start, std::size_t max_work,
    std::size_t k, std::size_t n) {
    const std::size_t wi = blockIdx.y;
    if (wi >= max_work) return;
    const std::uint32_t expert = work_expert[wi];
    if (expert >= apss26::NUM_EXPERTS) return;
    const std::size_t start = work_start[wi];
    const std::size_t end = offsets[expert + 1];
    __shared__ __align__(16) float As[2][BIG_BM][BIG_BK];
    __shared__ __align__(16) float Bs[2][WIDE_BN][BIG_BK];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BIG_BX + tx;
    const std::size_t block_col =
        static_cast<std::size_t>(blockIdx.x) * WIDE_BN;
    const std::size_t row0 = start + ty * BIG_TM;
    const std::size_t col0 = block_col + tx * WIDE_TN;
    float acc[BIG_TM][WIDE_TN] = {};
    const float* weight = weights[expert];
    const std::size_t tiles = k / BIG_BK;

    auto issue = [&](int stage, std::size_t p0) {
        constexpr int vectors_per_tile = BIG_BM * BIG_BK / 4;
        for (int vec = tid; vec < vectors_per_tile;
             vec += BIG_BX * BIG_BY) {
            const int lr = vec / (BIG_BK / 4);
            const int chunk = vec % (BIG_BK / 4);
            const std::size_t row = start + lr;
            const float* src = row < end
                ? a + row * k + p0 + chunk * 4 : a;
            cp_async_16(&As[stage][lr][chunk * 4], src,
                        row < end ? 16 : 0);
        }
        constexpr int b_vectors = WIDE_BN * BIG_BK / 4;
        for (int vec = tid; vec < b_vectors; vec += BIG_BX * BIG_BY) {
            const int lc = vec / (BIG_BK / 4);
            const int chunk = vec % (BIG_BK / 4);
            const int swizzled = chunk ^ ((lc >> 1) & 7);
            const std::size_t col = block_col + lc;
            const float* src = col < n
                ? weight + col * k + p0 + chunk * 4 : weight;
            cp_async_16(&Bs[stage][lc][swizzled * 4], src,
                        col < n ? 16 : 0);
        }
        cp_async_commit();
    };

    issue(0, 0);
    for (std::size_t tile = 0; tile < tiles; ++tile) {
        const int stage = tile & 1;
        if (tile + 1 < tiles) {
            issue(stage ^ 1, (tile + 1) * BIG_BK);
            cp_async_wait_one();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BIG_BK; ++p) {
            float bv[WIDE_TN];
#pragma unroll
            for (int cc = 0; cc < WIDE_TN; ++cc) {
                const int lc = tx * WIDE_TN + cc;
                const int physical_p =
                    ((p / 4) ^ ((lc >> 1) & 7)) * 4 + (p & 3);
                bv[cc] = Bs[stage][lc][physical_p];
            }
#pragma unroll
            for (int r = 0; r < BIG_TM; ++r) {
                const float av = As[stage][ty * BIG_TM + r][p];
#pragma unroll
                for (int cc = 0; cc < WIDE_TN; ++cc)
                    acc[r][cc] += av * bv[cc];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < BIG_TM; ++r)
#pragma unroll
        for (int cc = 0; cc < WIDE_TN; ++cc) {
            const std::size_t row = row0 + r, col = col0 + cc;
            if (row < end && col < n) out[row * n + col] = acc[r][cc];
        }
}

__global__ void k_moe_combine(
    const float* expert_out, const std::uint8_t* routes,
    const std::uint32_t* route_positions, float* out,
    std::size_t rows, std::size_t h) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * h) return;
    const std::size_t t = i / h, d = i % h;
    const std::uint8_t e0 = routes[2 * t], e1 = routes[2 * t + 1];
    const std::uint32_t p0 = route_positions[2 * t];
    const std::uint32_t p1 = route_positions[2 * t + 1];
    float value = 0.0f;
    if (e0 < e1) {
        value += 0.5f * expert_out[static_cast<std::size_t>(p0) * h + d];
        value += 0.5f * expert_out[static_cast<std::size_t>(p1) * h + d];
    } else {
        value += 0.5f * expert_out[static_cast<std::size_t>(p1) * h + d];
        value += 0.5f * expert_out[static_cast<std::size_t>(p0) * h + d];
    }
    out[i] = value;
}

__global__ void k_add(float* a, const float* b, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void k_add_bias(
    float* a, const float* bias, std::size_t n, std::size_t h) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += bias[i % h];
}

__global__ void k_silu_mul(
    const float* gate, const float* up, float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = gate[i];
        const float activated =
            x / (1.0f + static_cast<float>(exp(static_cast<double>(-x))));
        out[i] = activated * up[i];
    }
}

__global__ void k_layer_norm(
    const float* x, const float* weight, const float* bias, float* y,
    std::size_t h, float eps) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * h;
    extern __shared__ float staged[];
    __shared__ float mean;
    __shared__ float inv;
    for (std::size_t j = threadIdx.x; j < h; j += blockDim.x)
        staged[j] = x[base + j];
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < h; ++j) sum += staged[j];
        mean = sum / static_cast<float>(h);
        float var = 0.0f;
        for (std::size_t j = 0; j < h; ++j) {
            const float d = staged[j] - mean;
            var += d * d;
        }
        inv = 1.0f / sqrtf(var / static_cast<float>(h) + eps);
    }
    __syncthreads();
    for (std::size_t j = threadIdx.x; j < h; j += blockDim.x)
        y[base + j] =
            (staged[j] - mean) * inv * weight[j] + bias[j];
}

__global__ void k_add_layer_norm(
    float* x, const float* residual, const float* weight, const float* bias,
    float* y, std::size_t h, float eps) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * h;
    extern __shared__ float staged[];
    for (std::size_t j = threadIdx.x; j < h; j += blockDim.x) {
        const float value = x[base + j] + residual[base + j];
        x[base + j] = value;
        staged[j] = value;
    }
    __syncthreads();

    __shared__ float mean;
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < h; ++j) sum += staged[j];
        mean = sum / static_cast<float>(h);
        float var = 0.0f;
        for (std::size_t j = 0; j < h; ++j) {
            const float d = staged[j] - mean;
            var += d * d;
        }
        inv = 1.0f / sqrtf(var / static_cast<float>(h) + eps);
    }
    __syncthreads();
    for (std::size_t j = threadIdx.x; j < h; j += blockDim.x)
        y[base + j] =
            (staged[j] - mean) * inv * weight[j] + bias[j];
}

__global__ void k_rope(
    float* x, const std::uint32_t* row_pos,
    const float* cos_table, const float* sin_table,
    std::size_t rows, std::size_t heads, std::size_t head_dim) {
    const std::size_t half = head_dim / 2;
    const std::size_t pair =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= rows * heads * half) return;
    const std::size_t j = pair % half;
    const std::size_t head = (pair / half) % heads;
    const std::size_t row = pair / (half * heads);
    const std::size_t base = (row * heads + head) * head_dim;
    const std::size_t table = static_cast<std::size_t>(row_pos[row]) * half + j;
    const float c = cos_table[table], s = sin_table[table];
    const float x0 = x[base + j], x1 = x[base + j + half];
    x[base + j] = x0 * c - x1 * s;
    x[base + j + half] = x1 * c + x0 * s;
}

// One key slot per four threads. 16 keys x 4 heads keeps the staged tile
// small enough that occupancy stays ahead of the barrier saving; 32 was
// measured slower because the extra shared memory cost more residency than
// the deeper tile bought.
constexpr std::size_t ATTN_KEY_TILE = 16;

__global__ void k_attention(
    const float* q, const float* k, const float* v, float* out,
    const std::uint32_t* seg_offset, const std::uint32_t* seg_pos,
    std::size_t rows, std::size_t q_heads, std::size_t kv_heads,
    std::size_t head_dim, std::size_t max_seq) {
    const std::size_t task = blockIdx.x;
    const std::size_t row = task / q_heads;
    const std::size_t qh = task % q_heads;
    if (row >= rows) return;
    const std::size_t kh = qh / (q_heads / kv_heads);
    const std::size_t pos = seg_pos[row];
    const std::size_t begin =
        pos + 1 > apss26::SLIDING_WINDOW
            ? pos + 1 - apss26::SLIDING_WINDOW : 0;
    const std::size_t window = pos - begin + 1;
    extern __shared__ float shared[];
    float* scores = shared;
    float* query = scores + max_seq;
    float* key = query + head_dim;
    const std::size_t qbase = (row * q_heads + qh) * head_dim;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    if (threadIdx.x < head_dim)
        query[threadIdx.x] = q[qbase + threadIdx.x];
    __syncthreads();
    for (std::size_t w = 0; w < window; ++w) {
        const std::size_t key_row =
            static_cast<std::size_t>(seg_offset[row]) + begin + w;
        const std::size_t kbase =
            (key_row * kv_heads + kh) * head_dim;
        if (threadIdx.x < head_dim)
            key[threadIdx.x] = k[kbase + threadIdx.x];
        __syncthreads();
        if (threadIdx.x == 0) {
            float score = 0.0f;
            for (std::size_t d = 0; d < head_dim; ++d)
                score += query[d] * key[d];
            scores[w] = score * scale;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float maxv = -3.402823466e+38F;
        for (std::size_t w = 0; w < window; ++w)
            maxv = fmaxf(maxv, scores[w]);
        float denom = 0.0f;
        for (std::size_t w = 0; w < window; ++w) {
            scores[w] = static_cast<float>(
                exp(static_cast<double>(scores[w] - maxv)));
            denom += scores[w];
        }
        for (std::size_t w = 0; w < window; ++w) scores[w] /= denom;
    }
    __syncthreads();
    if (threadIdx.x < head_dim) {
        float value = 0.0f;
        for (std::size_t w = 0; w < window; ++w) {
            const std::size_t key_row =
                static_cast<std::size_t>(seg_offset[row]) + begin + w;
            const std::size_t vbase =
                (key_row * kv_heads + kh) * head_dim;
            value += scores[w] * v[vbase + threadIdx.x];
        }
        out[qbase + threadIdx.x] = value;
    }
}

// Phi GQA has four query heads per KV head. Group those query heads in one
// block so each key vector and value vector is fetched once instead of four
// times. The reduction order within every individual head remains d=0..127
// for QK and key=0..window-1 for the weighted value sum.
//
// The four heads of a group are independent until the value combine, so one
// thread per head runs the score and softmax passes instead of thread 0
// running all four back to back. Each head keeps its own sequential FP32
// accumulation order, so the results stay bit-identical. Queries and scores
// are stored head-minor so the four active threads land in four different
// shared-memory banks.
__global__ void k_attention_grouped_gqa4(
    const float* q, const float* k, const float* v, float* out,
    const std::uint32_t* seg_offset, const std::uint32_t* seg_pos,
    std::size_t rows, std::size_t q_heads, std::size_t kv_heads,
    std::size_t head_dim, std::size_t max_seq) {
    constexpr std::size_t group = 4;
    const std::size_t task = blockIdx.x;
    const std::size_t row = task / kv_heads;
    const std::size_t kh = task % kv_heads;
    if (row >= rows) return;
    const std::size_t pos = seg_pos[row];
    const std::size_t begin =
        pos + 1 > apss26::SLIDING_WINDOW
            ? pos + 1 - apss26::SLIDING_WINDOW : 0;
    const std::size_t window = pos - begin + 1;
    extern __shared__ float shared[];
    // scores[w * group + g] and queries[d * group + g] keep the group index
    // fastest so the score threads never share a shared-memory bank. keys is
    // a tile of ATTN_KEY_TILE staged key vectors, row-padded by one float so
    // the tile's rows land in different banks.
    const std::size_t key_stride = head_dim + 1;
    float* scores = shared;
    float* queries = scores + group * max_seq;
    float* keys = queries + group * head_dim;
    const std::size_t qh0 = kh * group;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    for (std::size_t i = threadIdx.x; i < group * head_dim;
         i += blockDim.x) {
        const std::size_t g = i / head_dim;
        const std::size_t d = i % head_dim;
        const std::size_t qbase = (row * q_heads + qh0 + g) * head_dim;
        queries[d * group + g] = q[qbase + d];
    }
    __syncthreads();

    // Scores for different keys are independent until the softmax, so run a
    // whole tile of them at once: thread t owns head t % group of key slot
    // t / group. Each score is still accumulated by a single thread over
    // d = 0..head_dim-1 in order, so the values are unchanged; what drops is
    // the barrier count, from two per key to two per tile.
    const std::size_t g = threadIdx.x % group;
    const std::size_t slot = threadIdx.x / group;
    const float* my_query = queries + g;
    for (std::size_t base = 0; base < window; base += ATTN_KEY_TILE) {
        const std::size_t count = window - base < ATTN_KEY_TILE
            ? window - base : ATTN_KEY_TILE;
        for (std::size_t i = threadIdx.x; i < count * head_dim;
             i += blockDim.x) {
            const std::size_t ks = i / head_dim;
            const std::size_t d = i % head_dim;
            const std::size_t key_row =
                static_cast<std::size_t>(seg_offset[row]) + begin + base + ks;
            const std::size_t kbase = (key_row * kv_heads + kh) * head_dim;
            keys[ks * key_stride + d] = k[kbase + d];
        }
        __syncthreads();
        if (slot < count) {
            const float* my_key = keys + slot * key_stride;
            float score = 0.0f;
            for (std::size_t d = 0; d < head_dim; ++d)
                score += my_query[d * group] * my_key[d];
            scores[(base + slot) * group + g] = score * scale;
        }
        __syncthreads();
    }

    if (threadIdx.x < group) {
        const std::size_t gi = threadIdx.x;
        float maxv = -3.402823466e+38F;
        for (std::size_t w = 0; w < window; ++w)
            maxv = fmaxf(maxv, scores[w * group + gi]);
        float denom = 0.0f;
        for (std::size_t w = 0; w < window; ++w) {
            const float e = static_cast<float>(
                exp(static_cast<double>(scores[w * group + gi] - maxv)));
            scores[w * group + gi] = e;
            denom += e;
        }
        for (std::size_t w = 0; w < window; ++w)
            scores[w * group + gi] /= denom;
    }
    __syncthreads();

    if (threadIdx.x < head_dim) {
        float value[group] = {};
        for (std::size_t w = 0; w < window; ++w) {
            const std::size_t key_row =
                static_cast<std::size_t>(seg_offset[row]) + begin + w;
            const std::size_t vbase =
                (key_row * kv_heads + kh) * head_dim;
            const float vv = v[vbase + threadIdx.x];
            const float* ws = scores + w * group;
#pragma unroll
            for (std::size_t gi = 0; gi < group; ++gi)
                value[gi] += ws[gi] * vv;
        }
#pragma unroll
        for (std::size_t gi = 0; gi < group; ++gi) {
            const std::size_t qbase =
                (row * q_heads + qh0 + gi) * head_dim;
            out[qbase + threadIdx.x] = value[gi];
        }
    }
}

__global__ void k_gather_rows(
    const float* x, const std::uint32_t* rows, float* out,
    std::size_t count, std::size_t h) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count * h)
        out[i] = x[static_cast<std::size_t>(rows[i / h]) * h + i % h];
}

__global__ void k_scatter_add_rows(
    const float* x, const std::uint32_t* rows, float scale, float* out,
    std::size_t count, std::size_t h) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count * h)
        out[static_cast<std::size_t>(rows[i / h]) * h + i % h] +=
            scale * x[i];
}

struct PackedMeta {
    std::vector<std::size_t> lens;
    std::uint32_t* offset = nullptr;
    std::uint32_t* pos = nullptr;
    std::size_t rows = 0;
    std::size_t max_seq = 0;
};

PackedMeta& packed_meta(const std::vector<std::size_t>& lens) {
    static PackedMeta meta;
    if (meta.lens == lens) return meta;
    if (meta.offset) cudaFree(meta.offset);
    if (meta.pos) cudaFree(meta.pos);
    meta = PackedMeta{};
    meta.lens = lens;
    for (std::size_t len : lens) {
        meta.rows += len;
        meta.max_seq = std::max(meta.max_seq, len);
    }
    std::vector<std::uint32_t> offset(meta.rows), pos(meta.rows);
    std::size_t base = 0;
    for (std::size_t len : lens) {
        for (std::size_t p = 0; p < len; ++p) {
            offset[base + p] = static_cast<std::uint32_t>(base);
            pos[base + p] = static_cast<std::uint32_t>(p);
        }
        base += len;
    }
    check_cuda(cudaMalloc(&meta.offset, meta.rows * sizeof(std::uint32_t)),
               "cudaMalloc packed offset");
    check_cuda(cudaMalloc(&meta.pos, meta.rows * sizeof(std::uint32_t)),
               "cudaMalloc packed pos");
    check_cuda(cudaMemcpy(meta.offset, offset.data(),
                          meta.rows * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "copy packed offset");
    check_cuda(cudaMemcpy(meta.pos, pos.data(),
                          meta.rows * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "copy packed pos");
    return meta;
}

struct RopeTable {
    float* cosine = nullptr;
    float* sine = nullptr;
    std::size_t max_seq = 0;
    std::size_t half = 0;
    float theta = 0.0f;
};

RopeTable& rope_table(
    std::size_t max_seq, std::size_t head_dim, float theta) {
    static RopeTable table;
    const std::size_t half = head_dim / 2;
    if (table.cosine && table.max_seq >= max_seq &&
        table.half == half && table.theta == theta) return table;
    if (table.cosine) cudaFree(table.cosine);
    if (table.sine) cudaFree(table.sine);
    table = RopeTable{};
    table.max_seq = max_seq;
    table.half = half;
    table.theta = theta;
    std::vector<float> cosine(max_seq * half), sine(max_seq * half);
    for (std::size_t p = 0; p < max_seq; ++p)
        for (std::size_t j = 0; j < half; ++j) {
            const float inv = std::pow(
                theta, -2.0f * static_cast<float>(j) /
                           static_cast<float>(head_dim));
            cosine[p * half + j] =
                std::cos(static_cast<float>(p) * inv);
            sine[p * half + j] =
                std::sin(static_cast<float>(p) * inv);
        }
    check_cuda(cudaMalloc(&table.cosine, cosine.size() * sizeof(float)),
               "cudaMalloc rope cosine");
    check_cuda(cudaMalloc(&table.sine, sine.size() * sizeof(float)),
               "cudaMalloc rope sine");
    check_cuda(cudaMemcpy(table.cosine, cosine.data(),
                          cosine.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "copy rope cosine");
    check_cuda(cudaMemcpy(table.sine, sine.data(),
                          sine.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "copy rope sine");
    return table;
}

std::uint32_t* device_rows(const std::vector<std::size_t>& rows) {
    static std::uint32_t* ptr = nullptr;
    static std::size_t capacity = 0;
    if (capacity < rows.size()) {
        if (ptr) cudaFree(ptr);
        check_cuda(cudaMalloc(&ptr, rows.size() * sizeof(std::uint32_t)),
                   "cudaMalloc row indices");
        capacity = rows.size();
    }
    std::vector<std::uint32_t> compact(rows.begin(), rows.end());
    check_cuda(cudaMemcpy(ptr, compact.data(),
                          compact.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "copy row indices");
    return ptr;
}

}

std::size_t register_moe_weights_gpu(
    const std::vector<const Tensor*>& w1,
    const std::vector<const Tensor*>& w2,
    const std::vector<const Tensor*>& w3) {
    if (w1.size() != apss26::NUM_EXPERTS ||
        w2.size() != apss26::NUM_EXPERTS ||
        w3.size() != apss26::NUM_EXPERTS)
        throw std::invalid_argument("MoE weight table size");
    std::vector<const float*> h_w1(apss26::NUM_EXPERTS);
    std::vector<const float*> h_w2(apss26::NUM_EXPERTS);
    std::vector<const float*> h_w3(apss26::NUM_EXPERTS);
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
        h_w1[e] = w1[e]->cuda_data();
        h_w2[e] = w2[e]->cuda_data();
        h_w3[e] = w3[e]->cuda_data();
    }
    MoeWeightSet set;
    const std::size_t bytes =
        apss26::NUM_EXPERTS * sizeof(const float*);
    check_cuda(cudaMalloc(&set.w1, bytes), "cudaMalloc MoE W1 table");
    check_cuda(cudaMalloc(&set.w2, bytes), "cudaMalloc MoE W2 table");
    check_cuda(cudaMalloc(&set.w3, bytes), "cudaMalloc MoE W3 table");
    check_cuda(cudaMemcpy(set.w1, h_w1.data(), bytes,
                          cudaMemcpyHostToDevice), "copy MoE W1 table");
    check_cuda(cudaMemcpy(set.w2, h_w2.data(), bytes,
                          cudaMemcpyHostToDevice), "copy MoE W2 table");
    check_cuda(cudaMemcpy(set.w3, h_w3.data(), bytes,
                          cudaMemcpyHostToDevice), "copy MoE W3 table");
    auto& registry = moe_weight_registry();
    registry.push_back(set);
    return registry.size();
}

void moe_forward_grouped_gpu(const Tensor& x, const Tensor& router,
                             std::size_t weights_handle, Tensor& out) {
    const std::size_t rows = x.size(0), h = x.size(1);
    if (router.size(0) != rows ||
        router.size(1) != apss26::NUM_EXPERTS ||
        weights_handle == 0 ||
        weights_handle > moe_weight_registry().size())
        throw std::invalid_argument("grouped MoE shape or handle");
    MoeScratch& scratch = moe_scratch(rows);
    const std::size_t assignments = 2 * rows;
    const std::size_t max_work =
        (assignments + BIG_BM - 1) / BIG_BM + apss26::NUM_EXPERTS;
    check_cuda(cudaMemsetAsync(scratch.work_expert, 0xff,
                               max_work * sizeof(std::uint32_t), 0),
               "clear MoE work queue");
    check_cuda(cudaMemsetAsync(scratch.counts, 0,
                               apss26::NUM_EXPERTS * sizeof(std::uint32_t), 0),
               "clear MoE counts");
    k_moe_top2<<<(rows + 255) / 256, 256>>>(
        router.cuda_data(), scratch.routes, scratch.counts, rows);
    k_moe_offsets_work<<<1, 32>>>(
        scratch.counts, scratch.offsets,
        scratch.work_expert, scratch.work_start);
    k_moe_fill_stable<<<apss26::NUM_EXPERTS, 256>>>(
        scratch.routes, scratch.offsets, scratch.assignment_rows,
        scratch.route_positions, rows);
    check_cuda(cudaGetLastError(), "MoE routing kernels launch");

    Tensor compact_input({assignments, h});
    const std::size_t input_elems = assignments * h;
    k_moe_gather_compact<<<(input_elems + 255) / 256, 256>>>(
        x.cuda_data(), scratch.assignment_rows,
        compact_input.cuda_data_write(), assignments, h);

    Tensor activated({assignments, apss26::EXPERT_INTERMEDIATE_SIZE});
    const MoeWeightSet& weights =
        moe_weight_registry()[weights_handle - 1];
    const dim3 block(BIG_BX, BIG_BY);
    const dim3 pair_grid(
        static_cast<unsigned>((apss26::EXPERT_INTERMEDIATE_SIZE +
                               BIG_BN - 1) / BIG_BN),
        static_cast<unsigned>(max_work));
    k_moe_pair_silu_grouped<<<pair_grid, block>>>(
        compact_input.cuda_data(), weights.w1, weights.w3,
        activated.cuda_data_write(), scratch.offsets,
        scratch.work_expert, scratch.work_start, max_work,
        h, apss26::EXPERT_INTERMEDIATE_SIZE);

    Tensor expert_output({assignments, h});
    const dim3 out_grid(
        static_cast<unsigned>((h + BIG_BN - 1) / BIG_BN),
        static_cast<unsigned>(max_work));
    if (apss26::EXPERT_INTERMEDIATE_SIZE % BIG_BK == 0 && h % WIDE_BN == 0) {
        static const bool wide_carveout = [] {
            prefer_shared_carveout(reinterpret_cast<const void*>(
                k_moe_matmul_grouped_wide_async));
            return true;
        }();
        (void)wide_carveout;
        const dim3 wide_grid(
            static_cast<unsigned>(h / WIDE_BN),
            static_cast<unsigned>(max_work));
        k_moe_matmul_grouped_wide_async<<<wide_grid, block>>>(
            activated.cuda_data(), weights.w2,
            expert_output.cuda_data_write(),
            scratch.offsets, scratch.work_expert, scratch.work_start,
            max_work, apss26::EXPERT_INTERMEDIATE_SIZE, h);
    } else if (apss26::EXPERT_INTERMEDIATE_SIZE % BIG_BK == 0) {
        static const bool carveout_set = [] {
            prefer_shared_carveout(reinterpret_cast<const void*>(
                k_moe_matmul_grouped_async));
            return true;
        }();
        (void)carveout_set;
        k_moe_matmul_grouped_async<<<out_grid, block>>>(
            activated.cuda_data(), weights.w2,
            expert_output.cuda_data_write(),
            scratch.offsets, scratch.work_expert, scratch.work_start,
            max_work, apss26::EXPERT_INTERMEDIATE_SIZE, h);
    } else {
        k_moe_matmul_grouped<<<out_grid, block>>>(
            activated.cuda_data(), weights.w2,
            expert_output.cuda_data_write(),
            scratch.offsets, scratch.work_expert, scratch.work_start,
            max_work, apss26::EXPERT_INTERMEDIATE_SIZE, h);
    }

    out = Tensor({rows, h});
    const std::size_t out_elems = rows * h;
    k_moe_combine<<<(out_elems + 255) / 256, 256>>>(
        expert_output.cuda_data(), scratch.routes,
        scratch.route_positions, out.cuda_data_write(), rows, h);
    check_cuda(cudaGetLastError(), "grouped MoE kernels launch");
}

void matmul_transposed_gpu(const Tensor& a, const Tensor& b, Tensor& c) {
    const std::size_t k = a.size(a.ndim() - 1);
    const std::size_t m = a.size() / k;
    const std::size_t n = b.size(0);
    if (b.size(1) != k || c.size(0) != m || c.size(1) != n)
        throw std::invalid_argument("matmul_transposed_gpu shape");
    const float* da = a.cuda_data();
    float* dc = c.cuda_data_write();
#ifdef USE_TC
    if (b.has_cuda_bf16_weight() && n % 64 == 0 && k % 16 == 0) {
        const dim3 grid(static_cast<unsigned>((n + 63) / 64),
                        static_cast<unsigned>((m + 63) / 64));
        k_matmul_transposed_bf16_wmma<<<grid, 256>>>(
            da,
            static_cast<const __nv_bfloat16*>(b.cuda_bf16_weight_hi()),
            static_cast<const __nv_bfloat16*>(b.cuda_bf16_weight_lo()),
            dc, m, k, n);
        check_cuda(cudaGetLastError(),
                   "k_matmul_transposed_bf16_wmma launch");
        return;
    }
#endif
    const float* db = b.cuda_data();
    if (m >= BIG_BM && n >= BIG_BN) {
        const dim3 block(BIG_BX, BIG_BY);
        const dim3 grid(
            static_cast<unsigned>((n + BIG_BN - 1) / BIG_BN),
            static_cast<unsigned>((m + BIG_BM - 1) / BIG_BM));
        if (k % BIG_BK == 0 && n % WIDE_BN == 0) {
            static const bool wide_carveout = [] {
                prefer_shared_carveout(reinterpret_cast<const void*>(
                    k_matmul_transposed_wide_async));
                return true;
            }();
            (void)wide_carveout;
            const dim3 wide_grid(
                static_cast<unsigned>(n / WIDE_BN),
                static_cast<unsigned>((m + BIG_BM - 1) / BIG_BM));
            k_matmul_transposed_wide_async<<<wide_grid, block>>>(
                da, db, dc, m, k, n);
            check_cuda(cudaGetLastError(),
                       "k_matmul_transposed_wide_async launch");
        } else if (k % BIG_BK == 0) {
            static const bool carveout_set = [] {
                prefer_shared_carveout(reinterpret_cast<const void*>(
                    k_matmul_transposed_register_blocked_async));
                return true;
            }();
            (void)carveout_set;
            k_matmul_transposed_register_blocked_async<<<grid, block>>>(
                da, db, dc, m, k, n);
            check_cuda(cudaGetLastError(),
                       "k_matmul_transposed_register_blocked_async launch");
        } else {
            k_matmul_transposed_register_blocked<<<grid, block>>>(
                da, db, dc, m, k, n);
            check_cuda(cudaGetLastError(),
                       "k_matmul_transposed_register_blocked launch");
        }
    } else {
        const dim3 block(TILE, TILE);
        const dim3 grid(
            static_cast<unsigned>((n + TILE - 1) / TILE),
            static_cast<unsigned>((m + TILE - 1) / TILE));
        k_matmul_transposed_tiled<<<grid, block>>>(
            da, db, dc, m, k, n);
        check_cuda(cudaGetLastError(), "k_matmul_transposed_tiled launch");
    }
}

void matmul_pair_silu_gpu(const Tensor& a, const Tensor& gate_weight,
                          const Tensor& up_weight, Tensor& out) {
    const std::size_t k = a.size(a.ndim() - 1);
    const std::size_t m = a.size() / k;
    const std::size_t n = gate_weight.size(0);
    if (gate_weight.size(1) != k || up_weight.size(0) != n ||
        up_weight.size(1) != k || out.size(0) != m || out.size(1) != n)
        throw std::invalid_argument("matmul_pair_silu_gpu shape");
    if (m < BIG_BM || n < BIG_BN) {
        Tensor gate({m, n}), up({m, n});
        matmul_transposed_gpu(a, gate_weight, gate);
        matmul_transposed_gpu(a, up_weight, up);
        silu_mul_gpu(gate, up, out);
        return;
    }
    const dim3 block(BIG_BX, BIG_BY);
    const dim3 grid(
        static_cast<unsigned>((n + BIG_BN - 1) / BIG_BN),
        static_cast<unsigned>((m + BIG_BM - 1) / BIG_BM));
    k_matmul_pair_silu_register_blocked<<<grid, block>>>(
        a.cuda_data(), gate_weight.cuda_data(), up_weight.cuda_data(),
        out.cuda_data_write(), m, k, n);
    check_cuda(cudaGetLastError(),
               "k_matmul_pair_silu_register_blocked launch");
}

void matmul_pair_bias_gpu(const Tensor& a,
                          const Tensor& first_weight,
                          const Tensor& first_bias, Tensor& first_out,
                          const Tensor& second_weight,
                          const Tensor& second_bias, Tensor& second_out) {
    const std::size_t k = a.size(a.ndim() - 1);
    const std::size_t m = a.size() / k;
    const std::size_t n = first_weight.size(0);
    if (first_weight.size(1) != k || second_weight.size(0) != n ||
        second_weight.size(1) != k || first_bias.size() != n ||
        second_bias.size() != n || first_out.size(0) != m ||
        first_out.size(1) != n || second_out.size(0) != m ||
        second_out.size(1) != n)
        throw std::invalid_argument("matmul_pair_bias_gpu shape");
    if (m < BIG_BM || n < BIG_BN) {
        matmul_transposed_gpu(a, first_weight, first_out);
        add_bias_inplace_gpu(first_out, first_bias);
        matmul_transposed_gpu(a, second_weight, second_out);
        add_bias_inplace_gpu(second_out, second_bias);
        return;
    }
    const dim3 block(BIG_BX, BIG_BY);
    if (n % WIDE_BN == 0) {
        static const bool wide_carveout = [] {
            prefer_shared_carveout(reinterpret_cast<const void*>(
                k_matmul_pair_bias_wide));
            return true;
        }();
        (void)wide_carveout;
        const dim3 wide_grid(
            static_cast<unsigned>(n / WIDE_BN),
            static_cast<unsigned>((m + BIG_BM - 1) / BIG_BM));
        k_matmul_pair_bias_wide<<<wide_grid, block>>>(
            a.cuda_data(), first_weight.cuda_data(), first_bias.cuda_data(),
            first_out.cuda_data_write(), second_weight.cuda_data(),
            second_bias.cuda_data(), second_out.cuda_data_write(), m, k, n);
        check_cuda(cudaGetLastError(), "k_matmul_pair_bias_wide launch");
        return;
    }
    const dim3 grid(
        static_cast<unsigned>((n + BIG_BN - 1) / BIG_BN),
        static_cast<unsigned>((m + BIG_BM - 1) / BIG_BM));
    k_matmul_pair_bias_register_blocked<<<grid, block>>>(
        a.cuda_data(), first_weight.cuda_data(), first_bias.cuda_data(),
        first_out.cuda_data_write(), second_weight.cuda_data(),
        second_bias.cuda_data(), second_out.cuda_data_write(), m, k, n);
    check_cuda(cudaGetLastError(),
               "k_matmul_pair_bias_register_blocked launch");
}

void add_inplace_gpu(Tensor& a, const Tensor& b) {
    if (a.size() != b.size()) throw std::invalid_argument("add shape");
    a.cuda_data();
    k_add<<<(a.size() + 255) / 256, 256>>>(
        a.cuda_data_write(), b.cuda_data(), a.size());
    check_cuda(cudaGetLastError(), "k_add launch");
}

void add_bias_inplace_gpu(Tensor& a, const Tensor& bias) {
    const std::size_t h = bias.size(0);
    if (bias.ndim() != 1 || a.size(a.ndim() - 1) != h)
        throw std::invalid_argument("bias shape");
    a.cuda_data();
    k_add_bias<<<(a.size() + 255) / 256, 256>>>(
        a.cuda_data_write(), bias.cuda_data(), a.size(), h);
    check_cuda(cudaGetLastError(), "k_add_bias launch");
}

void silu_mul_gpu(const Tensor& gate, const Tensor& up, Tensor& out) {
    if (gate.size() != up.size() || out.size() != gate.size())
        throw std::invalid_argument("silu_mul shape");
    k_silu_mul<<<(gate.size() + 255) / 256, 256>>>(
        gate.cuda_data(), up.cuda_data(), out.cuda_data_write(), gate.size());
    check_cuda(cudaGetLastError(), "k_silu_mul launch");
}

void layer_norm_gpu(const Tensor& x, const Tensor& weight, const Tensor& bias,
                    float eps, Tensor& y) {
    const std::size_t h = x.size(x.ndim() - 1);
    const std::size_t rows = x.size() / h;
    if (weight.size() != h || bias.size() != h || y.size() != x.size())
        throw std::invalid_argument("layer norm shape");
    k_layer_norm<<<rows, 256, h * sizeof(float)>>>(
        x.cuda_data(), weight.cuda_data(), bias.cuda_data(),
        y.cuda_data_write(), h, eps);
    check_cuda(cudaGetLastError(), "k_layer_norm launch");
}

void add_layer_norm_gpu(Tensor& x, const Tensor& residual,
                        const Tensor& weight, const Tensor& bias,
                        float eps, Tensor& y) {
    const std::size_t h = x.size(x.ndim() - 1);
    const std::size_t rows = x.size() / h;
    if (residual.size() != x.size() || weight.size() != h ||
        bias.size() != h || y.size() != x.size())
        throw std::invalid_argument("add layer norm shape");
    x.cuda_data();
    k_add_layer_norm<<<rows, 256, h * sizeof(float)>>>(
        x.cuda_data_write(), residual.cuda_data(), weight.cuda_data(),
        bias.cuda_data(), y.cuda_data_write(), h, eps);
    check_cuda(cudaGetLastError(), "k_add_layer_norm launch");
}

void apply_rope_gpu(Tensor& q, Tensor& k,
                    const std::vector<std::size_t>& seq_lens,
                    std::size_t q_heads, std::size_t kv_heads,
                    std::size_t head_dim, float theta) {
    PackedMeta& meta = packed_meta(seq_lens);
    RopeTable& table = rope_table(meta.max_seq, head_dim, theta);
    q.cuda_data();
    k.cuda_data();
    const std::size_t q_pairs = meta.rows * q_heads * (head_dim / 2);
    const std::size_t k_pairs = meta.rows * kv_heads * (head_dim / 2);
    k_rope<<<(q_pairs + 255) / 256, 256>>>(
        q.cuda_data_write(), meta.pos, table.cosine, table.sine,
        meta.rows, q_heads, head_dim);
    k_rope<<<(k_pairs + 255) / 256, 256>>>(
        k.cuda_data_write(), meta.pos, table.cosine, table.sine,
        meta.rows, kv_heads, head_dim);
    check_cuda(cudaGetLastError(), "k_rope launch");
}

void attention_gpu(const Tensor& q, const Tensor& k, const Tensor& v,
                   Tensor& out, const std::vector<std::size_t>& seq_lens,
                   std::size_t q_heads, std::size_t kv_heads,
                   std::size_t head_dim) {
    PackedMeta& meta = packed_meta(seq_lens);
    if (q_heads == 4 * kv_heads) {
        const std::size_t shared =
            (4 * meta.max_seq + 4 * head_dim +
             ATTN_KEY_TILE * (head_dim + 1)) * sizeof(float);
        k_attention_grouped_gqa4<<<meta.rows * kv_heads, 128, shared>>>(
            q.cuda_data(), k.cuda_data(), v.cuda_data(), out.cuda_data_write(),
            meta.offset, meta.pos, meta.rows, q_heads, kv_heads, head_dim,
            meta.max_seq);
        check_cuda(cudaGetLastError(),
                   "k_attention_grouped_gqa4 launch");
    } else {
        const std::size_t shared =
            (meta.max_seq + 2 * head_dim) * sizeof(float);
        k_attention<<<meta.rows * q_heads, 128, shared>>>(
            q.cuda_data(), k.cuda_data(), v.cuda_data(),
            out.cuda_data_write(), meta.offset, meta.pos, meta.rows,
            q_heads, kv_heads, head_dim, meta.max_seq);
        check_cuda(cudaGetLastError(), "k_attention launch");
    }
}

void gather_rows_gpu(const Tensor& x, const std::vector<std::size_t>& rows,
                     Tensor& out) {
    const std::size_t h = x.size(x.ndim() - 1);
    const std::size_t total = rows.size() * h;
    k_gather_rows<<<(total + 255) / 256, 256>>>(
        x.cuda_data(), device_rows(rows), out.cuda_data_write(),
        rows.size(), h);
    check_cuda(cudaGetLastError(), "k_gather_rows launch");
}

void scatter_add_rows_gpu(const Tensor& x,
                          const std::vector<std::size_t>& rows,
                          float scale, Tensor& out) {
    const std::size_t h = x.size(x.ndim() - 1);
    const std::size_t total = rows.size() * h;
    out.cuda_data();
    k_scatter_add_rows<<<(total + 255) / 256, 256>>>(
        x.cuda_data(), device_rows(rows), scale, out.cuda_data_write(),
        rows.size(), h);
    check_cuda(cudaGetLastError(), "k_scatter_add_rows launch");
}
}
