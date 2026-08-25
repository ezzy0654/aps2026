#include "tensor.h"
#include "config.h"
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace profiling {
namespace {
struct Stat {
    double total_seconds = 0.0;
    std::size_t calls = 0;
};
std::mutex g_mutex;
std::unordered_map<std::string, Stat> g_stats;
}  // namespace

bool enabled() {
    static const bool value = [] {
        const char* v = std::getenv("APS_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return value;
}

void reset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_stats.clear();
}

void record(const char* name, double seconds) {
    std::lock_guard<std::mutex> lock(g_mutex);
    Stat& stat = g_stats[name];
    stat.total_seconds += seconds;
    stat.calls += 1;
}

void report() {
    if (!enabled()) return;
    std::vector<std::pair<std::string, Stat>> rows;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        rows.assign(g_stats.begin(), g_stats.end());
    }
    if (rows.empty()) return;
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.second.total_seconds > b.second.total_seconds;
    });
    const double denom = rows.front().second.total_seconds;

    std::fprintf(stderr, "\n[APS_PROFILE] ---- section breakdown ----\n");
    for (const auto& [name, stat] : rows) {
        const double avg_ms = stat.total_seconds / static_cast<double>(stat.calls) * 1.0e3;
        const double pct = denom > 0.0 ? stat.total_seconds / denom * 100.0 : 0.0;
        std::fprintf(stderr,
                     "  %-24s total=%9.4f s  calls=%8zu  avg=%9.4f ms  (%5.1f%% of top)\n",
                     name.c_str(), stat.total_seconds, stat.calls, avg_ms, pct);
    }
    std::fprintf(stderr,
                 "[APS_PROFILE] note: sections are hierarchical - a parent's total\n"
                 "  already includes time spent inside its children.\n");
}

}  // namespace profiling

namespace {
void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
    }
}

// A handful of independently-growable device scratch slots, reused across
// tensor_ops calls so activations don't pay a cudaMalloc/cudaFree per call.
// Every tensor_ops function fully uploads-computes-downloads before
// returning (host code is single-threaded, one default stream), so slots
// can be shared across unrelated calls without any lifetime conflict.
enum ScratchSlot { SLOT_A = 0, SLOT_B, SLOT_C, SLOT_D, SLOT_E, SLOT_COUNT };

struct ScratchBuffer {
    float* ptr = nullptr;
    std::size_t capacity_elements = 0;
};
ScratchBuffer g_scratch[SLOT_COUNT];

float* scratch(int slot, std::size_t elements) {
    ScratchBuffer& buf = g_scratch[slot];
    if (buf.capacity_elements < elements) {
        if (buf.ptr) cudaFree(buf.ptr);
        cuda_check(cudaMalloc(&buf.ptr, elements * sizeof(float)), "scratch cudaMalloc");
        buf.capacity_elements = elements;
    }
    return buf.ptr;
}

// Same grow-on-demand idea as ScratchBuffer/scratch() above, generalized
// to a fixed number of independently-sized named slots (used by the
// device-resident MoE pipeline, where several buffers of different roles
// must stay alive across a whole PhiMoE::forward call).
template <int N>
struct DeviceBufferBank {
    ScratchBuffer entries[N];
    float* get(int slot, std::size_t elements) {
        ScratchBuffer& buf = entries[slot];
        if (buf.capacity_elements < elements) {
            if (buf.ptr) cudaFree(buf.ptr);
            cuda_check(cudaMalloc(&buf.ptr, elements * sizeof(float)), "DeviceBufferBank cudaMalloc");
            buf.capacity_elements = elements;
        }
        return buf.ptr;
    }
};
}  // namespace

Tensor::Tensor(std::vector<std::size_t> shape) : shape_(std::move(shape)) {
    std::size_t n = 1;
    for (std::size_t d : shape_) n *= d;
    data_.assign(n, 0.0f);
}

Tensor::~Tensor() { free_device(); }

void Tensor::free_device() noexcept {
    if (device_ptr_) {
        cudaFree(device_ptr_);
        device_ptr_ = nullptr;
        device_bytes_ = 0;
    }
}

Tensor::Tensor(const Tensor& other) : shape_(other.shape_), data_(other.data_) {
    if (other.device_ptr_) {
        cuda_check(cudaMalloc(&device_ptr_, other.device_bytes_), "Tensor copy cudaMalloc");
        device_bytes_ = other.device_bytes_;
        cuda_check(cudaMemcpy(device_ptr_, other.device_ptr_, device_bytes_, cudaMemcpyDeviceToDevice),
                   "Tensor copy D2D");
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    shape_ = other.shape_;
    data_ = other.data_;
    free_device();
    if (other.device_ptr_) {
        cuda_check(cudaMalloc(&device_ptr_, other.device_bytes_), "Tensor assign cudaMalloc");
        device_bytes_ = other.device_bytes_;
        cuda_check(cudaMemcpy(device_ptr_, other.device_ptr_, device_bytes_, cudaMemcpyDeviceToDevice),
                   "Tensor assign D2D");
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), data_(std::move(other.data_)),
      device_ptr_(other.device_ptr_), device_bytes_(other.device_bytes_) {
    other.device_ptr_ = nullptr;
    other.device_bytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
    free_device();
    device_ptr_ = other.device_ptr_;
    device_bytes_ = other.device_bytes_;
    other.device_ptr_ = nullptr;
    other.device_bytes_ = 0;
    return *this;
}

void Tensor::to_device() {
    const std::size_t bytes = data_.size() * sizeof(float);
    if (device_bytes_ != bytes) {
        free_device();
        cuda_check(cudaMalloc(&device_ptr_, bytes), "to_device cudaMalloc");
        device_bytes_ = bytes;
    }
    cuda_check(cudaMemcpy(device_ptr_, data_.data(), bytes, cudaMemcpyHostToDevice), "to_device H2D");
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

float& Tensor::at(std::size_t i) { return data_[offset({i})]; }
float& Tensor::at(std::size_t i, std::size_t j) { return data_[offset({i, j})]; }
float& Tensor::at(std::size_t i, std::size_t j, std::size_t k) { return data_[offset({i, j, k})]; }
float& Tensor::at(std::size_t i, std::size_t j, std::size_t k, std::size_t l) { return data_[offset({i, j, k, l})]; }
const float& Tensor::at(std::size_t i) const { return data_[offset({i})]; }
const float& Tensor::at(std::size_t i, std::size_t j) const { return data_[offset({i, j})]; }
const float& Tensor::at(std::size_t i, std::size_t j, std::size_t k) const { return data_[offset({i, j, k})]; }
const float& Tensor::at(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const { return data_[offset({i, j, k, l})]; }

void Tensor::reshape(std::vector<std::size_t> shape) {
    std::size_t n = 1;
    for (std::size_t d : shape) n *= d;
    if (n != data_.size()) throw std::invalid_argument("reshape changes tensor size");
    shape_ = std::move(shape);
}

void Tensor::fill(float value) { std::fill(data_.begin(), data_.end(), value); }

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------
namespace {

constexpr int kMatmulTile = 32;

// C[m,n] = sum_p A[m,p] * B[n,p]  (B is stored row-major as [N,K], i.e. the
// "transposed" operand already matches a Linear layer's weight layout).
//
// Compensated (Kahan) accumulation across tiles: each tile's 32 products sum
// into a local float, and those per-tile partials fold into `acc` with an
// error term carried along. Straight sequential accumulation over K=4096
// drifts enough through 32 decoder layers to push an occasional logit past
// the validation tolerance. The compensation runs once per tile, not per
// product, so the FLOP count is unchanged.
//
// `compensate` is a template parameter rather than a runtime flag so the
// plain path compiles down to exactly the original loop: the MoE gate must
// keep using it, because the router quantizes gate scores onto a 1e-3 grid
// and a *more* accurate gate actually disagrees with the reference's expert
// choice more often (measured: 3 routing flips in 305k decisions, each
// producing a much larger logit error than the drift this fixes).
// Register-blocked GEMM: C[M,N] = A[M,K] * B[N,K]^T, both operands
// K-contiguous.
//
// A block of 16x16 threads computes a 64x64 patch of C, so each thread owns
// a 4x4 micro-tile held in registers. Per step of the K loop a thread reads
// 4 values of A and 4 of B from shared memory and does 16 FMAs -- 2 FMAs per
// shared load, where the earlier one-output-per-thread kernel managed 0.5.
// Shared memory delivers 32 floats/cycle/SM against 128 FP32 lanes, so that
// ratio was the ceiling: it capped this kernel at ~12.5% of peak (measured
// 6.2%) and lifts it to ~50%.
//
// Crucially each acc[i][j] still accumulates p = 0..K-1 in order, exactly as
// the one-output-per-thread version did. Only the thread-to-output mapping
// changes, so results stay bit-identical -- which matters here, because the
// MoE router quantizes gate scores onto a 1e-3 grid and a different
// accumulation order can tip a token onto a different expert. Never split K
// across threads for the same reason.
constexpr int kBlockK = 16;
constexpr int kThreads = 256;
// Shared rows are padded so that (a) a row start stays 16B-aligned, which
// float4 loads require, and (b) the row stride stops being a multiple of 32,
// which is what made the transposing store below collide.
constexpr int kPad = 4;

// Tile geometry is a template parameter because the best choice depends on
// the shape. A 128x128 tile halves global traffic (each block covers 4x the
// output area, so A and B are re-read half as often), but it also produces
// 4x fewer blocks -- and the MoE experts, with M ~1238 and N=448, drop to 40
// blocks against 82 SMs. Measured: 128x128 wins 1.3-1.5x on the projections
// and *loses* 1.3x on the expert FFNs. launch_matmul picks per call.
//
// VEC vectorises both the global load and the shared read into float4. It
// needs K % 4 == 0 and 16B-aligned A/B, so launch_matmul checks and falls
// back to the scalar path otherwise; the two produce identical results.
template <int BM, int BN, int TM, int TN, bool VEC, int NT = kThreads>
__global__ __launch_bounds__(NT, 2) void matmul_transposed_blocked_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int M, int K, int N,
                                                 const tensor_ops::device::GroupTile* __restrict__ tiles,
                                                 long long weight_stride,
                                                 const float* __restrict__ bias) {
    static_assert(TM % 4 == 0 && TN % 4 == 0, "micro-tile must be float4-shaped");
    // Each thread's TM rows are split into TM/4 groups of 4 *contiguous*
    // rows, the groups spread BM/(TM/4) apart. The earlier layout gave a
    // thread 8 contiguous columns, i.e. a stride-8 shared read: within a warp
    // threadIdx.x = 0,4,8,12 then landed on the same bank at different
    // addresses, a 4-way conflict on the hottest instruction in the kernel.
    // At stride 4 a quarter-warp's eight float4 reads cover all 32 banks
    // exactly once, so the conflict disappears instead of merely shrinking.
    constexpr int VM = TM / 4;
    constexpr int VN = TN / 4;
    constexpr int LDA = BM + kPad;
    constexpr int LDB = BN + kPad;
    // Stored k-major so the inner loop reads consecutive threads' operands
    // from consecutive banks.
    // Two stages: the next K tile's global loads are issued before the current
    // tile is consumed, so the load latency overlaps the FMAs instead of
    // sitting in front of them. Costs one extra shared buffer (33.8 KB total,
    // still under the 48 KB static limit and still 2 blocks/SM, which
    // registers cap anyway) and AL+BL float4 of register staging.
    //
    // Note this is the pre-Ampere form. cp.async would carry no register cost,
    // but it can only copy contiguously, and this kernel's shared tiles are
    // k-major -- the store transposes. Matching cp.async would mean an m-major
    // shared layout, which is exactly what v11's conflict-free float4 inner
    // loop needs *not* to be.
    __shared__ float As[2][kBlockK][LDA];
    __shared__ float Bs[2][kBlockK][LDB];

    constexpr int AL = (BM * kBlockK) / (4 * NT);
    constexpr int BL = (BN * kBlockK) / (4 * NT);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;   // 0..255
    // Grouped mode (tiles != nullptr): several independent matmuls that share
    // K and N -- one per MoE expert -- run in a single launch. Each expert's
    // rows are contiguous in A, so a block only needs its row range and which
    // weight matrix to use; `M` becomes that expert's last row rather than the
    // buffer's. Whole-block-uniform, and outside the K loop, so it costs
    // nothing. The point is block count: one expert alone yields ~220 blocks
    // against the ~490 this GPU holds, and the 16 launches were serialised.
    int row0, m_end;
    if (tiles != nullptr) {
        const tensor_ops::device::GroupTile g = tiles[blockIdx.y];
        row0 = g.row0;
        m_end = g.row_end;
        B += static_cast<std::size_t>(g.weight_index) * static_cast<std::size_t>(weight_stride);
    } else {
        row0 = blockIdx.y * BM;
        m_end = M;
    }
    const int col0 = blockIdx.x * BN;
    const int a_row = threadIdx.y * 4;                        // this thread's rows
    const int b_col = threadIdx.x * 4;                        // this thread's cols

    float acc[TM][TN] = {};

    if constexpr (VEC) {
        constexpr int kVecK = kBlockK / 4;
        float4 ra[AL], rb[BL];


#define APS_ISSUE(k0)                                                                 \
        _Pragma("unroll")                                                             \
        for (int i = 0; i < AL; ++i) {                                                \
            const int lin = tid + i * NT, m = lin / kVecK, q = lin % kVecK;            \
            const int gr = row0 + m, gk = (k0) + q * 4;                                \
            ra[i] = (gr < m_end && gk + 3 < K)                                         \
                  ? *reinterpret_cast<const float4*>(A + static_cast<std::size_t>(gr) * K + gk) \
                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);                               \
        }                                                                             \
        _Pragma("unroll")                                                             \
        for (int i = 0; i < BL; ++i) {                                                \
            const int lin = tid + i * NT, n = lin / kVecK, q = lin % kVecK;            \
            const int gc = col0 + n, gk = (k0) + q * 4;                                \
            rb[i] = (gc < N && gk + 3 < K)                                             \
                  ? *reinterpret_cast<const float4*>(B + static_cast<std::size_t>(gc) * K + gk) \
                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);                               \
        }
#define APS_PARK(buf)                                                                 \
        _Pragma("unroll")                                                             \
        for (int i = 0; i < AL; ++i) {                                                \
            const int lin = tid + i * NT, m = lin / kVecK, q = lin % kVecK;            \
            As[buf][q * 4 + 0][m] = ra[i].x; As[buf][q * 4 + 1][m] = ra[i].y;          \
            As[buf][q * 4 + 2][m] = ra[i].z; As[buf][q * 4 + 3][m] = ra[i].w;          \
        }                                                                             \
        _Pragma("unroll")                                                             \
        for (int i = 0; i < BL; ++i) {                                                \
            const int lin = tid + i * NT, n = lin / kVecK, q = lin % kVecK;            \
            Bs[buf][q * 4 + 0][n] = rb[i].x; Bs[buf][q * 4 + 1][n] = rb[i].y;          \
            Bs[buf][q * 4 + 2][n] = rb[i].z; Bs[buf][q * 4 + 3][n] = rb[i].w;          \
        }

        APS_ISSUE(0)
        APS_PARK(0)
        __syncthreads();

        int cur = 0;
        for (int k0 = 0; k0 < K; k0 += kBlockK) {
            const int knext = k0 + kBlockK;
            if (knext < K) { APS_ISSUE(knext) }      // in flight during the FMAs below
#pragma unroll
            for (int p = 0; p < kBlockK; ++p) {
                float a[TM], b[TN];
#pragma unroll
                for (int f = 0; f < VM; ++f)
                    *reinterpret_cast<float4*>(&a[f * 4]) =
                        *reinterpret_cast<const float4*>(&As[cur][p][a_row + f * (BM / VM)]);
#pragma unroll
                for (int g = 0; g < VN; ++g)
                    *reinterpret_cast<float4*>(&b[g * 4]) =
                        *reinterpret_cast<const float4*>(&Bs[cur][p][b_col + g * (BN / VN)]);
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
            }
            // The buffer written here was last read one iteration ago, before
            // that iteration's barrier, so no thread is still in it.
            if (knext < K) { APS_PARK(1 - cur) }
            __syncthreads();
            cur ^= 1;
        }
#undef APS_ISSUE
#undef APS_PARK
    } else {
        for (int k0 = 0; k0 < K; k0 += kBlockK) {
#pragma unroll
            for (int i = 0; i < (BM * kBlockK) / NT; ++i) {
                const int lin = tid + i * NT;
                const int m = lin / kBlockK, p = lin % kBlockK;
                const int gr = row0 + m, gk = k0 + p;
                As[0][p][m] = (gr < m_end && gk < K) ? A[static_cast<std::size_t>(gr) * K + gk] : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < (BN * kBlockK) / NT; ++i) {
                const int lin = tid + i * NT;
                const int n = lin / kBlockK, p = lin % kBlockK;
                const int gc = col0 + n, gk = k0 + p;
                Bs[0][p][n] = (gc < N && gk < K) ? B[static_cast<std::size_t>(gc) * K + gk] : 0.0f;
            }
            __syncthreads();
#pragma unroll
            for (int p = 0; p < kBlockK; ++p) {
                float a[TM], b[TN];
#pragma unroll
                for (int f = 0; f < VM; ++f)
                    *reinterpret_cast<float4*>(&a[f * 4]) =
                        *reinterpret_cast<const float4*>(&As[0][p][a_row + f * (BM / VM)]);
#pragma unroll
                for (int g = 0; g < VN; ++g)
                    *reinterpret_cast<float4*>(&b[g * 4]) =
                        *reinterpret_cast<const float4*>(&Bs[0][p][b_col + g * (BN / VN)]);
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
            }
            __syncthreads();
        }
    }

    // This thread's TN bias entries depend only on (g, jj), not on the row, so
    // they are hoisted out of the row loop: read once instead of once per
    // stored element. Doing it inline cost 149 ms of GEMM time -- 8 distinct
    // values fetched 64 times each.
    float bv[TN];
    if (bias != nullptr) {
#pragma unroll
        for (int g = 0; g < VN; ++g)
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int gc = col0 + b_col + g * (BN / VN) + jj;
                bv[g * 4 + jj] = (gc < N) ? bias[gc] : 0.0f;
            }
    }

#pragma unroll
    for (int f = 0; f < VM; ++f) {
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
            const int gr = row0 + a_row + f * (BM / VM) + ii;
            if (gr >= m_end) continue;
#pragma unroll
            for (int g = 0; g < VN; ++g) {
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const int gc = col0 + b_col + g * (BN / VN) + jj;
                    // Bias folded into the store. It was a separate kernel that
                    // read C back and rewrote it -- 765 MB of traffic per layer
                    // to add an 8 KB vector. `bias` is block-uniform-null, and
                    // the no-bias path adds nothing at all (rather than adding
                    // 0.0f, which would turn a -0.0f accumulator into +0.0f).
                    if (gc < N)
                        C[static_cast<std::size_t>(gr) * N + gc] =
                            (bias != nullptr) ? acc[f * 4 + ii][g * 4 + jj] + bv[g * 4 + jj]
                                              : acc[f * 4 + ii][g * 4 + jj];
                }
            }
        }
    }
}

// Tall-skinny GEMM for a tiny N (the MoE gate is N = 16). Here the output is
// 1 MB against a 255 MB A, so the job is simply to stream A once at full
// bandwidth -- there is no reuse to tile for. The square-tiled kernel below is
// the wrong shape for it and measured 147 GB/s, 16% of peak.
//
// One block owns blockDim.y rows and all N columns; a K-chunk of both operands
// is staged in shared so A is read exactly once, coalesced. Each thread owns
// one (row, column) and walks p = 0..K-1 in order across the chunks, so the
// accumulation order is the reference's.
__global__ void matmul_narrow_n_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int K, int N,
                                       const float* __restrict__ bias) {
    constexpr int BK = 64;
    constexpr int LD = BK + 1;   // +1: the p-strided read below would otherwise
                                 // put every column on one bank.
    extern __shared__ float sh[];
    const int nrow = static_cast<int>(blockDim.y);
    float* As = sh;
    float* Bs = sh + nrow * LD;

    const int e = static_cast<int>(threadIdx.x);
    const int r = static_cast<int>(threadIdx.y);
    const int row0 = static_cast<int>(blockIdx.x) * nrow;
    const int tid = r * static_cast<int>(blockDim.x) + e;
    const int nt = static_cast<int>(blockDim.x * blockDim.y);

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < nrow * BK; idx += nt) {
            const int rr = idx / BK, cc = idx % BK;
            const int gr = row0 + rr, gk = k0 + cc;
            As[rr * LD + cc] = (gr < M && gk < K) ? A[static_cast<std::size_t>(gr) * K + gk] : 0.0f;
        }
        for (int idx = tid; idx < N * BK; idx += nt) {
            const int nn = idx / BK, cc = idx % BK;
            const int gk = k0 + cc;
            Bs[nn * LD + cc] = (gk < K) ? B[static_cast<std::size_t>(nn) * K + gk] : 0.0f;
        }
        __syncthreads();
        const int lim = (K - k0 < BK) ? (K - k0) : BK;
        for (int p = 0; p < lim; ++p) acc += As[r * LD + p] * Bs[e * LD + p];
        __syncthreads();
    }
    const int gr = row0 + r;
    if (gr < M && e < N)
        C[static_cast<std::size_t>(gr) * N + e] = (bias != nullptr) ? acc + bias[e] : acc;
}

// Narrow-N fallback (one output per thread, 32x32 tile). The blocked kernel
// above pads N up to 64, which would waste 3/4 of every block on the MoE
// gate (N=16); this wastes only half. Same K order, so the two agree bitwise.
__global__ void matmul_transposed_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int K, int N,
                                        const float* __restrict__ bias) {
    __shared__ float As[kMatmulTile][kMatmulTile];
    __shared__ float Bs[kMatmulTile][kMatmulTile];

    const int row = blockIdx.y * kMatmulTile + threadIdx.y;
    const int col = blockIdx.x * kMatmulTile + threadIdx.x;
    float acc = 0.0f;

    const int tiles = (K + kMatmulTile - 1) / kMatmulTile;
    for (int t = 0; t < tiles; ++t) {
        const int a_col = t * kMatmulTile + threadIdx.x;
        const int b_col = t * kMatmulTile + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[static_cast<std::size_t>(row) * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && b_col < K) ? B[static_cast<std::size_t>(col) * K + b_col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int p = 0; p < kMatmulTile; ++p) acc += As[threadIdx.y][p] * Bs[p][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[static_cast<std::size_t>(row) * N + col] = acc;
}

__global__ void add_inplace_kernel(float* a, const float* b, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void add_bias_inplace_kernel(float* a, const float* bias, std::size_t n, std::size_t h) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) a[i] += bias[i % h];
}

__global__ void mul_kernel(const float* a, const float* b, float* c, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

// Matches the CPU reference: silu(x) = x / (1 + exp(-x)), with exp evaluated
// in double precision before rounding back to float.
__global__ void silu_kernel(const float* x, float* y, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float e = static_cast<float>(exp(static_cast<double>(-x[i])));
        y[i] = x[i] / (1.0f + e);
    }
}

// Mean/variance accumulate in double: this output feeds the MoE gate, whose
// scores the router snaps onto a 1e-3 grid, so error here can tip a token
// onto a different expert than the reference chose. layer_norm is a small
// fraction of runtime, so the FP64 reductions are cheap insurance.
// One thread per row, accumulating in the same sequential FP32 order as the
// CPU reference. A parallel tree reduction is more accurate but rounds
// differently, and the answer file was produced with the sequential order --
// matching it matters more here than minimizing error, because the MoE
// router quantizes this output onto a 1e-3 grid and a last-place difference
// can tip a token onto a different expert.
// LayerNorm splits cleanly into a reduction and a map, and only the first
// half is order-constrained.
//
//   mean = (1/H) sum x_j            reduction -- FP addition is not
//   var  = (1/H) sum (x_j - mean)^2 associative, and the reference sums
//                                   sequentially, so this order is fixed
//                                   (see matching-reference-numerics.md)
//   y_j  = (x_j - mean) * inv * w_j + b_j     a pure map: y_j depends on
//                                   nothing but x_j, w_j, b_j and two
//                                   row constants
//
// The map was sharing a thread with the reduction purely because the CPU
// reference wrote them in one loop and the port copied its parallelisation
// axis (one core per row). That axis leaves the GPU 12% occupied -- there
// are only `rows` of them -- while the map has `rows * H` independent
// elements. It is also half the kernel's memory traffic. Splitting costs
// two floats per row.
__global__ void layer_norm_stats_kernel(const float* __restrict__ x, float eps,
                                        std::size_t h, std::size_t rows,
                                        float2* __restrict__ stats) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * h;

    float mean = 0.0f;
    for (std::size_t j = 0; j < h; ++j) mean += xr[j];
    mean /= static_cast<float>(h);
    float var = 0.0f;
    for (std::size_t j = 0; j < h; ++j) { const float d = xr[j] - mean; var += d * d; }
    stats[row] = make_float2(mean, 1.0f / sqrtf(var / static_cast<float>(h) + eps));
}

// One thread per element, so consecutive threads read consecutive x -- which
// also fixes the coalescing the row-per-thread form could not: there, a
// warp's 32 threads walked rows 16 KB apart.
__global__ void layer_norm_apply_kernel(const float* __restrict__ x,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        const float2* __restrict__ stats,
                                        float* __restrict__ y,
                                        std::size_t h, std::size_t rows) {
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= h) return;
    const float wj = weight[j];
    const float bj = bias[j];
    // gridDim.y is capped at 65535, so walk rows in that stride.
    for (std::size_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const float2 s = stats[row];
        const std::size_t o = row * h + j;
        y[o] = (x[o] - s.x) * s.y * wj + bj;
    }
}

// One block per row, with the row staged in shared memory. A row is h*4 bytes
// -- 16 KB at h = 4096 -- so it crosses the DRAM boundary exactly once: both
// reduction passes and the map all read it back out of shared.
//
// v16 split stats from apply, which fixed the map's coalescing but left the
// reduction as one thread per row walking 4 bytes at a time down a row whose
// neighbours are 16 KB away. That is 15,583 threads (~6 warps per SM of the 48
// available) each with about one outstanding miss, so roughly 63 KB is in
// flight against the 936 GB/s * ~600 ns ~= 562 KB needed to saturate the bus --
// measured 211 GB/s, 23% of peak. It is not a bandwidth problem, it is a
// memory-level-parallelism problem, and the fix is more independent loads in
// flight rather than a better access pattern.
//
// Staging also removes two of the four passes over x: stats read the row twice
// (mean, then variance) and apply read it a third time.
//
// The sums stay sequential in thread 0, in the reference's j = 0..h-1 order.
// Only where the bytes come from changes, so the result is bit-identical.
__global__ void layer_norm_fused_kernel(const float* __restrict__ x,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        float eps, float* __restrict__ y,
                                        int h, std::size_t rows) {
    extern __shared__ float s[];
    __shared__ float s_mean, s_inv;

    const std::size_t row = static_cast<std::size_t>(blockIdx.x);
    if (row >= rows) return;
    const std::size_t base = row * static_cast<std::size_t>(h);

    // float4 staging: the caller guarantees h % 4 == 0, and a row start is
    // row*h*4 bytes into a cudaMalloc'd buffer, so it is 16 B aligned.
    {
        const float4* __restrict__ xr4 = reinterpret_cast<const float4*>(x + base);
        float4* s4 = reinterpret_cast<float4*>(s);
        const int h4 = h >> 2;
        for (int i = static_cast<int>(threadIdx.x); i < h4; i += static_cast<int>(blockDim.x))
            s4[i] = xr4[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Both sums are read four-at-a-time so the shared-load latency overlaps
        // the add chain instead of sitting in front of every element. The adds
        // still happen in the reference's j = 0,1,2,... order -- only the loads
        // are batched -- so this is bit-identical to the scalar loop.
        const float4* __restrict__ s4 = reinterpret_cast<const float4*>(s);
        const int h4 = h >> 2;
        float mean = 0.0f;
        for (int i = 0; i < h4; ++i) {
            const float4 v = s4[i];
            mean += v.x; mean += v.y; mean += v.z; mean += v.w;
        }
        mean /= static_cast<float>(h);
        float var = 0.0f;
        for (int i = 0; i < h4; ++i) {
            const float4 v = s4[i];
            float d;
            d = v.x - mean; var += d * d;
            d = v.y - mean; var += d * d;
            d = v.z - mean; var += d * d;
            d = v.w - mean; var += d * d;
        }
        s_mean = mean;
        s_inv = 1.0f / sqrtf(var / static_cast<float>(h) + eps);
    }
    __syncthreads();

    const float mean = s_mean;
    const float inv = s_inv;
    for (int j = static_cast<int>(threadIdx.x); j < h; j += static_cast<int>(blockDim.x))
        y[base + j] = (s[j] - mean) * inv * weight[j] + bias[j];
}

// Grow-on-demand storage for the per-row (mean, inv) pair. Two floats per
// row -- 125 KB at this input's row count.
struct StatsBuffer { float2* ptr = nullptr; std::size_t capacity = 0; };
StatsBuffer g_ln_stats;

void launch_layer_norm(const float* x, const float* weight, const float* bias, float eps,
                       float* y, std::size_t rows, std::size_t h) {
    if (rows == 0 || h == 0) return;

    // Fused path: one block per row, row staged in shared. Needs h*4 bytes of
    // shared per block and a float4-aligned row stride; anything else (a huge
    // hidden size, an odd h) falls through to the split kernels below, which
    // produce identical results.
    const std::size_t shared_bytes = h * sizeof(float);
    if (h % 4 == 0 && shared_bytes <= 48u * 1024u &&
        rows <= static_cast<std::size_t>(2147483647)) {
        // Ask for the largest shared carveout so 16 KB blocks reach the
        // 6-blocks-per-SM the 1536-thread limit allows.
        static bool carveout_set = false;
        if (!carveout_set) {
            cudaFuncSetAttribute(reinterpret_cast<const void*>(layer_norm_fused_kernel),
                                 cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared);
            carveout_set = true;
        }
        layer_norm_fused_kernel<<<static_cast<unsigned>(rows), 256, shared_bytes>>>(
            x, weight, bias, eps, y, static_cast<int>(h), rows);
        cuda_check(cudaGetLastError(), "layer_norm_fused kernel");
        return;
    }

    if (g_ln_stats.capacity < rows) {
        if (g_ln_stats.ptr) cudaFree(g_ln_stats.ptr);
        cuda_check(cudaMalloc(&g_ln_stats.ptr, rows * sizeof(float2)), "layer_norm stats cudaMalloc");
        g_ln_stats.capacity = rows;
    }
    const int threads = 128;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    layer_norm_stats_kernel<<<blocks, threads>>>(x, eps, h, rows, g_ln_stats.ptr);

    const int athreads = 256;
    const unsigned gx = static_cast<unsigned>((h + athreads - 1) / athreads);
    const unsigned gy = static_cast<unsigned>(rows < 65535 ? rows : 65535);
    layer_norm_apply_kernel<<<dim3(gx, gy), athreads>>>(x, weight, bias, g_ln_stats.ptr, y, h, rows);
    cuda_check(cudaGetLastError(), "layer_norm kernel");
}

// The rotation angle depends on exactly two things -- the token's position
// within its own sequence, and which of the head_dim/2 coordinate pairs is
// being rotated. It does *not* depend on the data being rotated, nor on the
// head, nor on the layer. So there are only max_position * half distinct
// (cos, sin) pairs in the whole run: 32 * 64 = 2048 here, a 16KB table,
// against the ~811M evaluations a thread-per-(token, head, pair) launch
// performs. See rope_table_kernel below -- it carries the original
// expressions unchanged, so the floats it stores are the same bits this
// kernel used to compute.
__global__ void rope_kernel(float* qk, std::size_t seq_len, std::size_t heads,
                            std::size_t head_dim, const float2* __restrict__ table,
                            const int* __restrict__ position_of_row) {
    const std::size_t half = head_dim / 2;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = seq_len * heads * half;
    if (idx >= total) return;

    const std::size_t j = idx % half;
    const std::size_t h = (idx / half) % heads;
    const std::size_t s = idx / (half * heads);
    const std::size_t position = static_cast<std::size_t>(position_of_row[s]);

    // Consecutive threads take consecutive j, so this is one coalesced
    // 8-byte load out of a table small enough to sit in L1.
    const float2 cs = table[position * half + j];
    const float c = cs.x;
    const float sn = cs.y;

    float* base = qk + s * (heads * head_dim) + h * head_dim;
    const float x0 = base[j];
    const float x1 = base[j + half];
    base[j] = x0 * c - x1 * sn;
    base[j + half] = x1 * c + x0 * sn;
}

// One thread per (position, coordinate pair). The arithmetic below is
// character-for-character what rope_kernel used to run inline: the reference
// computes these with glibc's float pow/cos/sin, which are accurate to well
// under an ulp, where CUDA's powf/cosf/sinf are only ~2 ulp and that
// difference propagates into every q/k element. Evaluating in double and
// rounding once reproduces the correctly-rounded float result. The
// intermediate exponent and angle stay float, as they are on the host.
//
// Because the stored values are the same floats, and rope_kernel's rotation
// is unchanged, this is memoization rather than approximation -- the output
// is bit-identical.
__global__ void rope_table_kernel(float2* table, std::size_t max_positions,
                                  std::size_t half, std::size_t head_dim, float theta) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= max_positions * half) return;
    const std::size_t j = idx % half;
    const std::size_t position = idx / half;

    const float exponent = -2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
    const float inv = static_cast<float>(pow(static_cast<double>(theta), static_cast<double>(exponent)));
    const float angle = static_cast<float>(position) * inv;
    table[idx] = make_float2(static_cast<float>(cos(static_cast<double>(angle))),
                             static_cast<float>(sin(static_cast<double>(angle))));
}

// Grow-on-demand storage for the table above, rebuilt on every call. Caching
// it across calls would shave ~30us but would mean a warm-up run could
// populate it for a timed one, which the assignment's timing-integrity rule
// forbids; at this cost there is no reason to go near that line.
struct RopeTableBuffer { float2* ptr = nullptr; std::size_t capacity = 0; };
RopeTableBuffer g_rope_table;

const float2* ensure_rope_table(std::size_t max_positions, std::size_t head_dim, float theta) {
    const std::size_t half = head_dim / 2;
    const std::size_t entries = max_positions * half;
    if (entries == 0) return nullptr;
    if (g_rope_table.capacity < entries) {
        if (g_rope_table.ptr) cudaFree(g_rope_table.ptr);
        cuda_check(cudaMalloc(&g_rope_table.ptr, entries * sizeof(float2)), "rope table cudaMalloc");
        g_rope_table.capacity = entries;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((entries + threads - 1) / threads);
    rope_table_kernel<<<blocks, threads>>>(g_rope_table.ptr, max_positions, half, head_dim, theta);
    cuda_check(cudaGetLastError(), "rope_table kernel");
    return g_rope_table.ptr;
}

// One block per (query head, query position). Scores are computed once into
// shared memory, but everything after that reproduces the CPU reference's
// arithmetic exactly: `denom` is summed sequentially over the window, and the
// output accumulates `(e / denom) * v` per key -- dividing inside the loop,
// as the reference does, not once at the end. Those two choices change the
// FP32 rounding, and matching the answer file's rounding is what keeps a
// borderline MoE routing decision on the same side.
constexpr int kAttnThreads = 128;

__global__ void sliding_window_attention_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, int seq_len, int q_heads, int kv_heads, int head_dim, int window,
    const int* __restrict__ position_of_row, const int* __restrict__ path, int max_len) {
    extern __shared__ float shared_e[];

    const int qi = blockIdx.y;
    const int qh = blockIdx.x;
    const int group = q_heads / kv_heads;
    const int kh = qh / group;
    // Rows are prefix-trie nodes, so a query's keys are its ancestor chain
    // rather than the span of rows before it. Everything else -- the window
    // clamp, the causal cutoff, the order the keys are visited -- is the same
    // computation in position space instead of row space.
    const int pos = position_of_row[qi];
    const int window_lo = pos - window + 1;
    const int lo = (window_lo > 0) ? window_lo : 0;
    const int len = pos - lo + 1;
    const int* __restrict__ keys = path + static_cast<std::size_t>(qi) * max_len + lo;
    const float scale = sqrtf(static_cast<float>(head_dim));

    const float* qv = q + static_cast<std::size_t>(qi) * q_heads * head_dim + static_cast<std::size_t>(qh) * head_dim;

    for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
        const int ki = keys[idx];
        const float* kv_row = k + static_cast<std::size_t>(ki) * kv_heads * head_dim + static_cast<std::size_t>(kh) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) score += qv[d] * kv_row[d];
        shared_e[idx] = score / scale;
    }
    __syncthreads();

    // The reference walks max and denom sequentially, and `denom`'s addition
    // order has to be reproduced -- see matching-reference-numerics. But only
    // the *accumulation* is order-dependent: exp(e - maxv) depends on nothing
    // but that element and maxv, so whichever thread evaluates it produces the
    // same bits. Splitting the reference's single loop into three phases keeps
    // every ordered operation ordered while lifting the expensive one out of
    // the serial section. Measured: the double-precision exp was 44% of this
    // kernel, and one thread was doing all ~12 of them per block.
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        float maxv = -INFINITY;
        for (int idx = 0; idx < len; ++idx) maxv = fmaxf(maxv, shared_e[idx]);
        s_max = maxv;
    }
    __syncthreads();
    const float maxv = s_max;

    // Order-independent, so every thread takes a share.
    for (int idx = threadIdx.x; idx < len; idx += blockDim.x)
        shared_e[idx] = static_cast<float>(exp(static_cast<double>(shared_e[idx] - maxv)));
    __syncthreads();

    __shared__ float s_denom;
    if (threadIdx.x == 0) {
        float denom = 0.0f;
        for (int idx = 0; idx < len; ++idx) denom += shared_e[idx];
        s_denom = denom;
    }
    __syncthreads();
    const float denom = s_denom;

    float* outv = out + static_cast<std::size_t>(qi) * q_heads * head_dim + static_cast<std::size_t>(qh) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int idx = 0; idx < len; ++idx) {
            const int ki = keys[idx];
            acc += shared_e[idx] / denom *
                   v[static_cast<std::size_t>(ki) * kv_heads * head_dim + static_cast<std::size_t>(kh) * head_dim + d];
        }
        outv[d] = acc;
    }
}

// dst[i,:] = src[indices[i],:] — used to gather the tokens routed to one
// MoE expert directly from the device-resident layer activation, with no
// host round trip.
__global__ void gather_rows_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                   const int* __restrict__ indices, std::size_t num_rows, std::size_t row_width) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = num_rows * row_width;
    if (idx >= total) return;
    const std::size_t i = idx / row_width;
    const std::size_t j = idx % row_width;
    dst[idx] = src[static_cast<std::size_t>(indices[i]) * row_width + j];
}

// dst[indices[i],:] += weights[i] * src[i,:]. Within one launch every
// indices[i] is distinct (a token is routed to a given expert at most
// once), so no two threads ever touch the same dst element — plain += is
// race-free without atomics. Across experts' separate (sequential,
// same-stream) launches, a token's two picks land safely one after another.
__global__ void scatter_add_rows_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                        const int* __restrict__ indices, const float* __restrict__ weights,
                                        std::size_t num_rows, std::size_t row_width) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = num_rows * row_width;
    if (idx >= total) return;
    const std::size_t i = idx / row_width;
    const std::size_t j = idx % row_width;
    dst[static_cast<std::size_t>(indices[i]) * row_width + j] += weights[i] * src[idx];
}

// Dispatches on N: the register-blocked kernel pads N up to 64, so for the
// MoE gate (N=16) the older 32-wide kernel wastes less. Both keep the same
// K accumulation order and agree bitwise.
void launch_matmul(const float* d_a, const float* d_b, float* d_c,
                   std::size_t m, std::size_t k, std::size_t n, const char* what,
                   const float* d_bias = nullptr) {
    constexpr std::size_t kSMs = 82;                 // RTX 3090
    const auto blocks_for = [&](std::size_t bm, std::size_t bn) {
        return ((n + bn - 1) / bn) * ((m + bm - 1) / bm);
    };
    // float4 loads need every A/B row to start on a 16B boundary. Row r
    // starts at r*K floats, so K % 4 == 0 plus an aligned base is enough.
    // Every shape in this model qualifies (K is 4096, 2048 or 448); the
    // check exists so an odd K silently takes the scalar path instead of
    // faulting.
    const bool vec = (k % 4 == 0) &&
                     (reinterpret_cast<std::uintptr_t>(d_a) % 16 == 0) &&
                     (reinterpret_cast<std::uintptr_t>(d_b) % 16 == 0);
    if (n < static_cast<std::size_t>(kMatmulTile)) {
        // Narrow N (the MoE gate, N=16): tall-skinny, so stream A once rather
        // than tile it. Falls back to the old kernel if the row group's shared
        // tile would not fit; both produce identical results.
        const int nn = static_cast<int>(n);
        int nrow = 256 / nn; if (nrow > 32) nrow = 32; if (nrow < 1) nrow = 1;
        const std::size_t shb =
            static_cast<std::size_t>(nrow + nn) * 65 * sizeof(float);
        if (shb <= 48u * 1024u) {
            const dim3 block(static_cast<unsigned>(nn), static_cast<unsigned>(nrow));
            const dim3 grid(static_cast<unsigned>((m + nrow - 1) / nrow));
            matmul_narrow_n_kernel<<<grid, block, shb>>>(
                d_a, d_b, d_c, static_cast<int>(m), static_cast<int>(k), nn, d_bias);
            cuda_check(cudaGetLastError(), what);
            return;
        }
        const dim3 block(kMatmulTile, kMatmulTile);
        const dim3 grid(static_cast<unsigned>((n + kMatmulTile - 1) / kMatmulTile),
                        static_cast<unsigned>((m + kMatmulTile - 1) / kMatmulTile));
        matmul_transposed_kernel<<<grid, block>>>(
            d_a, d_b, d_c, static_cast<int>(m), static_cast<int>(k), static_cast<int>(n), d_bias);
    } else if (blocks_for(128, 128) >= 2 * kSMs) {
        // Measured 2026-08-25: a rectangular 128x256 or 256x128 tile (512
        // threads) keeps registers at 119 and threads/SM at 512 while raising
        // arithmetic intensity 32 -> 42.7, past the 38 flop/byte where this GPU
        // turns compute-bound, and it does cut traffic 1.34x as predicted. It
        // still lost by ~7%: one 512-thread block per SM means a
        // __syncthreads() stalls all 16 warps, where two 256-thread blocks
        // cover for each other. Achieved bandwidth fell 692 -> 489 GB/s, more
        // than the traffic saving. Tile geometry is therefore settled.
        const dim3 block(128 / 8, 128 / 8);          // 16x16 = 256
        const dim3 grid(static_cast<unsigned>((n + 127) / 128),
                        static_cast<unsigned>((m + 127) / 128));
        if (vec) matmul_transposed_blocked_kernel<128, 128, 8, 8, true><<<grid, block>>>(
                     d_a, d_b, d_c, (int)m, (int)k, (int)n, nullptr, 0, d_bias);
        else     matmul_transposed_blocked_kernel<128, 128, 8, 8, false><<<grid, block>>>(
                     d_a, d_b, d_c, (int)m, (int)k, (int)n, nullptr, 0, d_bias);
    } else {
        // Too few blocks at 128x128 (the MoE experts land here): a smaller
        // tile spreads the work over more SMs, which outweighs re-reading.
        const dim3 block(64 / 4, 64 / 4);            // 16x16 = 256
        const dim3 grid(static_cast<unsigned>((n + 63) / 64),
                        static_cast<unsigned>((m + 63) / 64));
        if (vec)
            matmul_transposed_blocked_kernel<64, 64, 4, 4, true><<<grid, block>>>(
                d_a, d_b, d_c, static_cast<int>(m), static_cast<int>(k), static_cast<int>(n), nullptr, 0, d_bias);
        else
            matmul_transposed_blocked_kernel<64, 64, 4, 4, false><<<grid, block>>>(
                d_a, d_b, d_c, static_cast<int>(m), static_cast<int>(k), static_cast<int>(n), nullptr, 0, d_bias);
    }
    cuda_check(cudaGetLastError(), what);
}

// Grouped counterpart. Tile geometry is passed in rather than derived, so the
// caller (which had to know BM to build the tile map) and the launch cannot
// disagree.
void launch_matmul_grouped(const float* d_a, const float* d_b, float* d_c,
                           const tensor_ops::device::GroupTile* d_tiles,
                           std::size_t num_tiles, std::size_t k, std::size_t n,
                           std::size_t weight_stride, std::size_t block_m,
                           const char* what) {
    if (num_tiles == 0) return;
    const bool vec = (k % 4 == 0) &&
                     (reinterpret_cast<std::uintptr_t>(d_a) % 16 == 0) &&
                     (reinterpret_cast<std::uintptr_t>(d_b) % 16 == 0);
    const int ki = static_cast<int>(k), ni = static_cast<int>(n);
    const long long stride = static_cast<long long>(weight_stride);
    if (block_m == 128) {
        const dim3 block(128 / 8, 128 / 8);
        const dim3 grid(static_cast<unsigned>((n + 127) / 128), static_cast<unsigned>(num_tiles));
        if (vec) matmul_transposed_blocked_kernel<128, 128, 8, 8, true><<<grid, block>>>(
                     d_a, d_b, d_c, 0, ki, ni, d_tiles, stride, nullptr);
        else     matmul_transposed_blocked_kernel<128, 128, 8, 8, false><<<grid, block>>>(
                     d_a, d_b, d_c, 0, ki, ni, d_tiles, stride, nullptr);
    } else {
        const dim3 block(64 / 4, 64 / 4);
        const dim3 grid(static_cast<unsigned>((n + 63) / 64), static_cast<unsigned>(num_tiles));
        if (vec) matmul_transposed_blocked_kernel<64, 64, 4, 4, true><<<grid, block>>>(
                     d_a, d_b, d_c, 0, ki, ni, d_tiles, stride, nullptr);
        else     matmul_transposed_blocked_kernel<64, 64, 4, 4, false><<<grid, block>>>(
                     d_a, d_b, d_c, 0, ki, ni, d_tiles, stride, nullptr);
    }
    cuda_check(cudaGetLastError(), what);
}
}  // namespace

namespace tensor_ops {

void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c) {
    APS_PROFILE_SCOPE("tensor.matmul_transposed");
    const std::size_t m = a.size(0), k = a.size(1), n = b.size(0);
    if (b.size(1) != k || c.size(0) != m || c.size(1) != n) throw std::invalid_argument("matmul_transposed shape");
    if (!b.device_data()) throw std::runtime_error("matmul_transposed: weight tensor is not resident on device");

    float* d_a = scratch(SLOT_A, a.size());
    float* d_c = scratch(SLOT_C, c.size());
    cuda_check(cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice), "matmul H2D a");
    launch_matmul(d_a, b.device_data(), d_c, m, k, n, "matmul_transposed kernel");
    cuda_check(cudaMemcpy(c.data(), d_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost), "matmul D2H c");
}

void add_inplace(Tensor& a, const Tensor& b) {
    APS_PROFILE_SCOPE("tensor.add_inplace");
    if (a.size() != b.size()) throw std::invalid_argument("add shape");

    float* d_a = scratch(SLOT_A, a.size());
    float* d_b = scratch(SLOT_B, b.size());
    cuda_check(cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice), "add H2D a");
    cuda_check(cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice), "add H2D b");
    const int threads = 256;
    const int blocks = static_cast<int>((a.size() + threads - 1) / threads);
    add_inplace_kernel<<<blocks, threads>>>(d_a, d_b, a.size());
    cuda_check(cudaGetLastError(), "add_inplace kernel");
    cuda_check(cudaMemcpy(a.data(), d_a, a.size() * sizeof(float), cudaMemcpyDeviceToHost), "add D2H a");
}

void add_bias_inplace(Tensor& a, const Tensor& bias) {
    APS_PROFILE_SCOPE("tensor.add_bias_inplace");
    if (bias.ndim() != 1 || a.size(a.ndim() - 1) != bias.size(0)) throw std::invalid_argument("bias shape");
    if (!bias.device_data()) throw std::runtime_error("add_bias_inplace: bias tensor is not resident on device");
    const std::size_t h = bias.size(0);

    float* d_a = scratch(SLOT_A, a.size());
    cuda_check(cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice), "bias H2D a");
    const int threads = 256;
    const int blocks = static_cast<int>((a.size() + threads - 1) / threads);
    add_bias_inplace_kernel<<<blocks, threads>>>(d_a, bias.device_data(), a.size(), h);
    cuda_check(cudaGetLastError(), "add_bias_inplace kernel");
    cuda_check(cudaMemcpy(a.data(), d_a, a.size() * sizeof(float), cudaMemcpyDeviceToHost), "bias D2H a");
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    APS_PROFILE_SCOPE("tensor.mul");
    if (a.size() != b.size() || c.size() != a.size()) throw std::invalid_argument("mul shape");

    float* d_a = scratch(SLOT_A, a.size());
    float* d_b = scratch(SLOT_B, b.size());
    float* d_c = scratch(SLOT_C, c.size());
    cuda_check(cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice), "mul H2D a");
    cuda_check(cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice), "mul H2D b");
    const int threads = 256;
    const int blocks = static_cast<int>((a.size() + threads - 1) / threads);
    mul_kernel<<<blocks, threads>>>(d_a, d_b, d_c, a.size());
    cuda_check(cudaGetLastError(), "mul kernel");
    cuda_check(cudaMemcpy(c.data(), d_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost), "mul D2H c");
}

void silu(const Tensor& x, Tensor& y) {
    APS_PROFILE_SCOPE("tensor.silu");
    if (x.size() != y.size()) throw std::invalid_argument("silu shape");

    float* d_x = scratch(SLOT_A, x.size());
    float* d_y = scratch(SLOT_C, y.size());
    cuda_check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "silu H2D x");
    const int threads = 256;
    const int blocks = static_cast<int>((x.size() + threads - 1) / threads);
    silu_kernel<<<blocks, threads>>>(d_x, d_y, x.size());
    cuda_check(cudaGetLastError(), "silu kernel");
    cuda_check(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost), "silu D2H y");
}

void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& y) {
    APS_PROFILE_SCOPE("tensor.layer_norm");
    const std::size_t h = x.size(x.ndim() - 1), rows = x.size() / h;
    if (weight.size() != h || bias.size() != h || y.size() != x.size()) throw std::invalid_argument("layer norm shape");
    if (!weight.device_data() || !bias.device_data())
        throw std::runtime_error("layer_norm: weight/bias tensor is not resident on device");

    float* d_x = scratch(SLOT_A, x.size());
    float* d_y = scratch(SLOT_C, y.size());
    cuda_check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "layer_norm H2D x");
    launch_layer_norm(d_x, weight.device_data(), bias.device_data(), eps, d_y, rows, h);
    cuda_check(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost), "layer_norm D2H y");
}

// Tensor-based entry point kept from the skeleton; seq_start_of_row is now
// read as a per-row position array (device pointer), matching the device
// variant below. Nothing on the hot path calls this.
void apply_rope(Tensor& q, Tensor& k, std::size_t seq_len,
                std::size_t q_heads, std::size_t kv_heads, std::size_t head_dim, float theta,
                const int* seq_start_of_row) {
    APS_PROFILE_SCOPE("tensor.apply_rope");
    float* d_q = scratch(SLOT_A, q.size());
    float* d_k = scratch(SLOT_B, k.size());
    cuda_check(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "rope H2D q");
    cuda_check(cudaMemcpy(d_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice), "rope H2D k");

    const std::size_t half = head_dim / 2;
    const int threads = 256;
    // position = row - seq_start_of_row[row] < seq_len, so seq_len bounds the
    // table. That is loose (the device variant below is handed the exact
    // maximum), but this entry point isn't on any hot path.
    const float2* table = ensure_rope_table(seq_len, head_dim, theta);
    {
        const std::size_t total = seq_len * q_heads * half;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        rope_kernel<<<blocks, threads>>>(d_q, seq_len, q_heads, head_dim, table, seq_start_of_row);
    }
    {
        const std::size_t total = seq_len * kv_heads * half;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        rope_kernel<<<blocks, threads>>>(d_k, seq_len, kv_heads, head_dim, table, seq_start_of_row);
    }
    cuda_check(cudaGetLastError(), "apply_rope kernel");
    cuda_check(cudaMemcpy(q.data(), d_q, q.size() * sizeof(float), cudaMemcpyDeviceToHost), "rope D2H q");
    cuda_check(cudaMemcpy(k.data(), d_k, k.size() * sizeof(float), cudaMemcpyDeviceToHost), "rope D2H k");
}

void sliding_window_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              std::size_t seq_len, std::size_t q_heads, std::size_t kv_heads,
                              std::size_t head_dim, std::size_t window,
                              const int* seq_start_of_row, Tensor& out) {
    (void)seq_start_of_row;
    APS_PROFILE_SCOPE("tensor.sliding_window_attention");
    float* d_q = scratch(SLOT_A, q.size());
    float* d_k = scratch(SLOT_B, k.size());
    float* d_v = scratch(SLOT_D, v.size());
    float* d_out = scratch(SLOT_E, out.size());
    cuda_check(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "attn H2D q");
    cuda_check(cudaMemcpy(d_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice), "attn H2D k");
    cuda_check(cudaMemcpy(d_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "attn H2D v");

    const dim3 grid(static_cast<unsigned>(q_heads), static_cast<unsigned>(seq_len));
    const std::size_t shared_bytes = window * sizeof(float);
    // The skeleton's Tensor form has no trie behind it: every row is its own
    // sequence position, so the path is the identity run 0..seq_len-1.
    std::vector<int> pos(seq_len), path(seq_len * seq_len);
    for (std::size_t r = 0; r < seq_len; ++r) {
        pos[r] = static_cast<int>(r);
        for (std::size_t j = 0; j <= r; ++j) path[r * seq_len + j] = static_cast<int>(j);
    }
    const device::RowMap rows = device::upload_row_map(pos.data(), path.data(), seq_len, seq_len);
    sliding_window_attention_kernel<<<grid, kAttnThreads, shared_bytes>>>(
        d_q, d_k, d_v, d_out, static_cast<int>(seq_len), static_cast<int>(q_heads),
        static_cast<int>(kv_heads), static_cast<int>(head_dim), static_cast<int>(window),
        rows.position, rows.path, rows.max_len);
    cuda_check(cudaGetLastError(), "sliding_window_attention kernel");
    cuda_check(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost), "attn D2H out");
}

namespace device {
namespace {
// Separate from the g_scratch[] pool above: these must stay alive across
// an entire PhiMoE::forward call (Flat/Y persist through the whole expert
// loop), not just for the duration of one tensor_ops call.
DeviceBufferBank<static_cast<int>(Buffer::Count)> g_moe_float;  // indexed by Buffer
struct IntBuffer { int* ptr = nullptr; std::size_t capacity_elements = 0; };
IntBuffer g_moe_indices;
DeviceBufferBank<1> g_moe_weights;
// Separate again from g_moe_indices: that one is deliberately overwritten
// per expert within a single forward_chunk call, while this one must stay
// valid across all 32 decoder layers of that same call.
IntBuffer g_seq_start;
IntBuffer g_row_path;
IntBuffer g_token_ids;
}  // namespace

void check(cudaError_t err, const char* what) { cuda_check(err, what); }

float* buffer(Buffer which, std::size_t elements) {
    return g_moe_float.get(static_cast<int>(which), elements);
}

int* index_buffer(std::size_t elements) {
    if (g_moe_indices.capacity_elements < elements) {
        if (g_moe_indices.ptr) cudaFree(g_moe_indices.ptr);
        cuda_check(cudaMalloc(&g_moe_indices.ptr, elements * sizeof(int)), "index_buffer cudaMalloc");
        g_moe_indices.capacity_elements = elements;
    }
    return g_moe_indices.ptr;
}

int* token_id_buffer(std::size_t elements) {
    if (g_token_ids.capacity_elements < elements) {
        if (g_token_ids.ptr) cudaFree(g_token_ids.ptr);
        cuda_check(cudaMalloc(&g_token_ids.ptr, elements * sizeof(int)), "token_id_buffer cudaMalloc");
        g_token_ids.capacity_elements = elements;
    }
    return g_token_ids.ptr;
}

float* weight_buffer(std::size_t elements) { return g_moe_weights.get(0, elements); }

RowMap upload_row_map(const int* position, const int* path,
                      std::size_t rows, std::size_t max_len) {
    if (g_seq_start.capacity_elements < rows) {
        if (g_seq_start.ptr) cudaFree(g_seq_start.ptr);
        cuda_check(cudaMalloc(&g_seq_start.ptr, rows * sizeof(int)), "row position cudaMalloc");
        g_seq_start.capacity_elements = rows;
    }
    const std::size_t path_n = rows * max_len;
    if (g_row_path.capacity_elements < path_n) {
        if (g_row_path.ptr) cudaFree(g_row_path.ptr);
        cuda_check(cudaMalloc(&g_row_path.ptr, path_n * sizeof(int)), "row path cudaMalloc");
        g_row_path.capacity_elements = path_n;
    }
    cuda_check(cudaMemcpy(g_seq_start.ptr, position, rows * sizeof(int), cudaMemcpyHostToDevice),
               "row position H2D");
    cuda_check(cudaMemcpy(g_row_path.ptr, path, path_n * sizeof(int), cudaMemcpyHostToDevice),
               "row path H2D");
    return RowMap{g_seq_start.ptr, g_row_path.ptr, static_cast<int>(max_len)};
}



void matmul_transposed(const float* d_a, const Tensor& weight, float* d_c, std::size_t m,
                       const Tensor* bias) {
    if (!weight.device_data()) throw std::runtime_error("device::matmul_transposed: weight not resident on device");
    const float* d_bias = (bias != nullptr && bias->size()) ? bias->device_data() : nullptr;
    if (d_bias == nullptr && bias != nullptr && bias->size())
        throw std::runtime_error("device::matmul_transposed: bias not resident on device");
    launch_matmul(d_a, weight.device_data(), d_c, m, weight.size(1), weight.size(0),
                  "device::matmul_transposed kernel", d_bias);
}

// Kept as a separate entry point only because PhiMoE's gate calls it; the
// compensated/uncompensated distinction it used to carry is gone (a Kahan
// variant was tried and reverted -- see the kernel comment above).
void matmul_transposed_uncompensated(const float* d_a, const Tensor& weight, float* d_c, std::size_t m) {
    matmul_transposed(d_a, weight, d_c, m);
}

void silu(const float* d_x, float* d_y, std::size_t n) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    silu_kernel<<<blocks, threads>>>(d_x, d_y, n);
    cuda_check(cudaGetLastError(), "device::silu kernel");
}

void mul(const float* d_a, const float* d_b, float* d_c, std::size_t n) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    mul_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cuda_check(cudaGetLastError(), "device::mul kernel");
}

void gather_rows(const float* d_src, float* d_dst, const int* d_indices,
                 std::size_t num_rows, std::size_t row_width) {
    const std::size_t total = num_rows * row_width;
    if (total == 0) return;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    gather_rows_kernel<<<blocks, threads>>>(d_src, d_dst, d_indices, num_rows, row_width);
    cuda_check(cudaGetLastError(), "gather_rows kernel");
}

void scatter_add_rows(float* d_dst, const float* d_src, const int* d_indices, const float* d_weights,
                      std::size_t num_rows, std::size_t row_width) {
    const std::size_t total = num_rows * row_width;
    if (total == 0) return;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    scatter_add_rows_kernel<<<blocks, threads>>>(d_dst, d_src, d_indices, d_weights, num_rows, row_width);
    cuda_check(cudaGetLastError(), "scatter_add_rows kernel");
}

void add_inplace(float* d_a, const float* d_b, std::size_t n) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    add_inplace_kernel<<<blocks, threads>>>(d_a, d_b, n);
    cuda_check(cudaGetLastError(), "device::add_inplace kernel");
}

void add_bias_inplace(float* d_a, const Tensor& bias, std::size_t n) {
    if (!bias.device_data()) throw std::runtime_error("device::add_bias_inplace: bias not resident on device");
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    add_bias_inplace_kernel<<<blocks, threads>>>(d_a, bias.device_data(), n, bias.size(0));
    cuda_check(cudaGetLastError(), "device::add_bias_inplace kernel");
}

void layer_norm(const float* d_x, const Tensor& weight, const Tensor& bias,
                float eps, float* d_y, std::size_t rows, std::size_t h) {
    if (!weight.device_data() || !bias.device_data())
        throw std::runtime_error("device::layer_norm: weight/bias not resident on device");
    launch_layer_norm(d_x, weight.device_data(), bias.device_data(), eps, d_y, rows, h);
}

void apply_rope(float* d_q, float* d_k, std::size_t seq_len, std::size_t q_heads,
                std::size_t kv_heads, std::size_t head_dim,
                const RowMap& rows, const float2* rope_table) {
    const std::size_t half = head_dim / 2;
    const int threads = 256;
    {
        const std::size_t total = seq_len * q_heads * half;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        rope_kernel<<<blocks, threads>>>(d_q, seq_len, q_heads, head_dim, rope_table, rows.position);
    }
    {
        const std::size_t total = seq_len * kv_heads * half;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        rope_kernel<<<blocks, threads>>>(d_k, seq_len, kv_heads, head_dim, rope_table, rows.position);
    }
    cuda_check(cudaGetLastError(), "device::apply_rope kernel");
}

const float2* build_rope_table(std::size_t max_positions, std::size_t head_dim, float theta) {
    return ensure_rope_table(max_positions, head_dim, theta);
}

namespace {
struct GroupTileBuffer { GroupTile* ptr = nullptr; std::size_t capacity = 0; };
GroupTileBuffer g_group_tiles;
}  // namespace

const GroupTile* upload_group_tiles(const GroupTile* host, std::size_t n) {
    if (n == 0) return nullptr;
    if (g_group_tiles.capacity < n) {
        if (g_group_tiles.ptr) cudaFree(g_group_tiles.ptr);
        cuda_check(cudaMalloc(&g_group_tiles.ptr, n * sizeof(GroupTile)), "group tiles cudaMalloc");
        g_group_tiles.capacity = n;
    }
    cuda_check(cudaMemcpy(g_group_tiles.ptr, host, n * sizeof(GroupTile), cudaMemcpyHostToDevice),
               "group tiles H2D");
    return g_group_tiles.ptr;
}

void matmul_transposed_grouped(const float* d_a, const Tensor& weights,
                               std::size_t experts, float* d_c,
                               const GroupTile* d_tiles, std::size_t num_tiles,
                               std::size_t block_m) {
    if (!weights.device_data())
        throw std::runtime_error("matmul_transposed_grouped: weights not resident on device");
    const std::size_t k = weights.size(1);
    const std::size_t n = weights.size(0) / experts;
    launch_matmul_grouped(d_a, weights.device_data(), d_c, d_tiles, num_tiles, k, n,
                          n * k, block_m, "matmul_transposed_grouped kernel");
}

void sliding_window_attention(const float* d_q, const float* d_k, const float* d_v,
                              std::size_t seq_len, std::size_t q_heads, std::size_t kv_heads,
                              std::size_t head_dim, std::size_t window,
                              const RowMap& rows, float* d_out) {
    const dim3 grid(static_cast<unsigned>(q_heads), static_cast<unsigned>(seq_len));
    const std::size_t shared_bytes = window * sizeof(float);
    sliding_window_attention_kernel<<<grid, kAttnThreads, shared_bytes>>>(
        d_q, d_k, d_v, d_out, static_cast<int>(seq_len), static_cast<int>(q_heads),
        static_cast<int>(kv_heads), static_cast<int>(head_dim), static_cast<int>(window),
        rows.position, rows.path, rows.max_len);
    cuda_check(cudaGetLastError(), "device::sliding_window_attention kernel");
}

}  // namespace device
}  // namespace tensor_ops
