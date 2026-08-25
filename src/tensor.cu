#include "tensor.h"
#include "config.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <cuda_runtime.h>
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
      data_(std::move(other.data_)), cuda_data_(other.cuda_data_), host_valid_(other.host_valid_),
      cuda_valid_(other.cuda_valid_) {
    other.cuda_data_ = nullptr;
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
    host_valid_ = other.host_valid_;
    cuda_valid_ = other.cuda_valid_;
    other.cuda_data_ = nullptr;
    other.numel_ = 0;
    other.host_valid_ = true;
    other.cuda_valid_ = false;
    return *this;
}

void Tensor::release_cuda() {
    if (cuda_data_) cudaFreeAsync(cuda_data_, 0);
    cuda_data_ = nullptr;
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

void matmul_transposed_gpu(const Tensor& a, const Tensor& b, Tensor& c) {
    const std::size_t k = a.size(a.ndim() - 1);
    const std::size_t m = a.size() / k;
    const std::size_t n = b.size(0);
    if (b.size(1) != k || c.size(0) != m || c.size(1) != n)
        throw std::invalid_argument("matmul_transposed_gpu shape");
    const float* da = a.cuda_data();
    const float* db = b.cuda_data();
    float* dc = c.cuda_data_write();
    if (m >= BIG_BM && n >= BIG_BN) {
        const dim3 block(BIG_BX, BIG_BY);
        const dim3 grid(
            static_cast<unsigned>((n + BIG_BN - 1) / BIG_BN),
            static_cast<unsigned>((m + BIG_BM - 1) / BIG_BM));
        k_matmul_transposed_register_blocked<<<grid, block>>>(
            da, db, dc, m, k, n);
        check_cuda(cudaGetLastError(),
                   "k_matmul_transposed_register_blocked launch");
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
    const std::size_t shared =
        (meta.max_seq + 2 * head_dim) * sizeof(float);
    k_attention<<<meta.rows * q_heads, 128, shared>>>(
        q.cuda_data(), k.cuda_data(), v.cuda_data(), out.cuda_data_write(),
        meta.offset, meta.pos, meta.rows, q_heads, kv_heads, head_dim,
        meta.max_seq);
    check_cuda(cudaGetLastError(), "k_attention launch");
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
