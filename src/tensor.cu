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
    // data_'s allocator makes resize() allocation-only (see
    // detail::DefaultInitAllocator in tensor.h), so this explicit fill is
    // what actually zero-initialises -- same values, same single-threaded
    // page-fault cost std::vector's own value-init would have paid. Every
    // existing caller of Tensor(shape) is therefore unaffected; only
    // uninitialized_parallel() below deliberately skips this.
    data_.resize(n);
    std::fill(data_.begin(), data_.end(), 0.0f);
}

// See the declaration in tensor.h for when this is (and is not) safe.
Tensor Tensor::uninitialized_parallel(std::vector<std::size_t> shape) {
    Tensor t;
    t.shape_ = std::move(shape);
    std::size_t n = 1;
    for (std::size_t d : t.shape_) n *= d;
    t.data_.resize(n);
    float* p = t.data_.data();
    const long ni = static_cast<long>(n);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < ni; ++i) p[i] = 0.0f;
    return t;
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
template <int BM, int BN, int TM, int TN, bool VEC, int NT = kThreads, int MINB = 2>
__global__ __launch_bounds__(NT, MINB) void matmul_transposed_blocked_kernel(const float* __restrict__ A,
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
                    for (int jj = 0; jj < TN; ++jj) {
                        // Serpentine: alternating rows walk b backwards so the
                        // operand-reuse cache keeps a[i]/b[j] alive across the
                        // row boundary instead of breaking at every one. Only
                        // the order in which *independent* accumulators are
                        // touched within one p changes; every acc[i][j] keeps
                        // its own ascending-p chain, so this is bit-identical.
                        const int j = (i & 1) ? (TN - 1 - jj) : jj;
                        acc[i][j] += a[i] * b[j];
                    }
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
                    for (int jj = 0; jj < TN; ++jj) {
                        // Serpentine: alternating rows walk b backwards so the
                        // operand-reuse cache keeps a[i]/b[j] alive across the
                        // row boundary instead of breaking at every one. Only
                        // the order in which *independent* accumulators are
                        // touched within one p changes; every acc[i][j] keeps
                        // its own ascending-p chain, so this is bit-identical.
                        const int j = (i & 1) ? (TN - 1 - jj) : jj;
                        acc[i][j] += a[i] * b[j];
                    }
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

// Fused silu(gate)*up epilogue for a w1||w3-concatenated grouped GEMM: gateup
// is [rows, 2*inter], row r's w1 half at [0,inter) and w3 half at
// [inter,2*inter). Same silu formula as silu_kernel above (exp in double),
// so bit-identical to running silu_kernel then mul_kernel as two passes --
// only the number of reads/writes of the 896-wide row changes.
__global__ void silu_mul_fused_kernel(const float* __restrict__ gateup,
                                      float* __restrict__ out,
                                      std::size_t rows, int inter) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = rows * static_cast<std::size_t>(inter);
    if (idx >= total) return;
    const std::size_t r = idx / static_cast<std::size_t>(inter);
    const int j = static_cast<int>(idx - r * static_cast<std::size_t>(inter));
    const std::size_t base = r * static_cast<std::size_t>(2 * inter);
    const float g = gateup[base + static_cast<std::size_t>(j)];
    const float u = gateup[base + static_cast<std::size_t>(inter + j)];
    const float e = static_cast<float>(exp(static_cast<double>(-g)));
    out[idx] = (g / (1.0f + e)) * u;
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

// One warp per row instead of one block per row. layer_norm_fused_kernel
// above stages a row whole (16 KB) so only ONE thread per block (lane/thread
// 0) ever does the serial FADD chain; the other 255 threads stage the row
// then sit parked at __syncthreads() until it finishes. Measured: 35 of 40
// resident warps blocked there, 5 blocks/SM (16 KB shared each), 0.49
// eligible warps/scheduler -- occupancy looks fine, almost nothing is
// actually eligible to issue.
//
// Here each of a block's 8 warps owns a different row and streams it through
// a 512-float shared tile (2 KB) rather than staging it whole, so 8 rows
// cost the same 16 KB/block the old kernel spent on 1 -- blocks/SM is
// unchanged, but each now runs 8 independent chains instead of 1 (~5x8=40
// resident chains against the old 5). __syncwarp() replaces
// __syncthreads(), so one warp's long chain never blocks the other 7.
//
// The price is reading the row twice more from DRAM (once per pass below,
// where the old kernel read it once): mean and variance still can't share a
// pass without changing the reduction to something order-dependent (a tree,
// or a single-pass Sum(x^2) formula) -- both already rejected, see
// matching-reference-numerics.md. Every accumulation is still one thread
// walking j = 0..h-1 in the reference's order, so results are bit-identical;
// only where the bytes come from, and how many other chains run alongside,
// changes.
__global__ void layer_norm_warp_kernel(const float* __restrict__ x,
                                       const float* __restrict__ weight,
                                       const float* __restrict__ bias,
                                       float eps, float* __restrict__ y,
                                       int h, std::size_t rows) {
    constexpr int kTile = 512;
    extern __shared__ float smem[];

    const int warp = static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    const std::size_t warps_per_block = blockDim.x >> 5;
    const std::size_t row = static_cast<std::size_t>(blockIdx.x) * warps_per_block + warp;
    if (row >= rows) return;

    float* tile = smem + static_cast<std::size_t>(warp) * kTile;
    const float* xr = x + row * static_cast<std::size_t>(h);
    const int ntiles = h / kTile;

    float mean = 0.0f;
    for (int t = 0; t < ntiles; ++t) {
        for (int i = lane; i < kTile; i += 32) tile[i] = xr[t * kTile + i];
        __syncwarp();
        if (lane == 0)
            for (int i = 0; i < kTile; ++i) mean += tile[i];
        __syncwarp();
    }
    if (lane == 0) mean /= static_cast<float>(h);
    mean = __shfl_sync(0xffffffffu, mean, 0);

    float var = 0.0f;
    for (int t = 0; t < ntiles; ++t) {
        for (int i = lane; i < kTile; i += 32) tile[i] = xr[t * kTile + i];
        __syncwarp();
        if (lane == 0)
            for (int i = 0; i < kTile; ++i) { const float d = tile[i] - mean; var += d * d; }
        __syncwarp();
    }
    float inv = 0.0f;
    if (lane == 0) inv = 1.0f / sqrtf(var / static_cast<float>(h) + eps);
    inv = __shfl_sync(0xffffffffu, inv, 0);

    float* yr = y + row * static_cast<std::size_t>(h);
    for (int j = lane; j < h; j += 32)
        yr[j] = (xr[j] - mean) * inv * weight[j] + bias[j];
}

// Grow-on-demand storage for the per-row (mean, inv) pair. Two floats per
// row -- 125 KB at this input's row count.
struct StatsBuffer { float2* ptr = nullptr; std::size_t capacity = 0; };
StatsBuffer g_ln_stats;

void launch_layer_norm(const float* x, const float* weight, const float* bias, float eps,
                       float* y, std::size_t rows, std::size_t h) {
    if (rows == 0 || h == 0) return;

    // Warp-per-row path: needs h to divide evenly into 512-float tiles (true
    // for this model's fixed h=4096) and blockIdx.x to cover all rows at 8
    // warps/block.
    constexpr int kTile = 512;
    constexpr int kWarpsPerBlock = 8;
    if (h % kTile == 0 &&
        rows <= static_cast<std::size_t>(2147483647) * kWarpsPerBlock) {
        static bool carveout_set = false;
        if (!carveout_set) {
            cudaFuncSetAttribute(reinterpret_cast<const void*>(layer_norm_warp_kernel),
                                 cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared);
            carveout_set = true;
        }
        const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
        const std::size_t shared_bytes =
            static_cast<std::size_t>(kWarpsPerBlock) * kTile * sizeof(float);
        layer_norm_warp_kernel<<<blocks, kWarpsPerBlock * 32, shared_bytes>>>(
            x, weight, bias, eps, y, static_cast<int>(h), rows);
        cuda_check(cudaGetLastError(), "layer_norm_warp kernel");
        return;
    }

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

// Warp-per-query attention for the short-key case. ncu put 52.8% of the
// general kernel's stall cycles on `barrier`: it runs five phases separated by
// four block-wide __syncthreads(), and two of those phases execute on
// threadIdx.x == 0 while the other 127 threads wait. The shape is right for
// the window it was written against (2047 keys spread over 128 threads) and
// wrong for the data, where a row's key list is capped by the trie at
// max_len = 32.
//
// Once len <= 32 the whole score vector fits in one register per lane, so a
// warp can carry a query from scores to output with no block-wide barrier at
// all -- warp-level lockstep replaces every one of them. A block is one GQA
// group (the q_heads sharing a kv head), which also puts the four warps that
// read identical K and V rows next to each other in L1.
//
// The order-dependent steps are untouched: the K dot product still runs
// d = 0..head_dim-1 inside a single lane, `denom` still sums idx = 0..len-1
// sequentially, and the AV loop still divides per key rather than by a hoisted
// reciprocal. Only the max scan changes shape, and fmaxf is associative and
// commutative, so the shuffle tree returns the bits the sequential scan did.
constexpr int kWarpAttnMaxLen = 32;   // lanes per warp: one score per lane
constexpr int kAttnGroup = 4;         // warps per block = q_heads / kv_heads
// K is staged 64 coordinates at a time. A full row would be 32*129*4 = 16.5 KB
// and cap the SM at 6 blocks against the 12 the warp budget allows; half a row
// is 8.3 KB and costs one block. The +1 is what makes the read conflict-free:
// at stride 64 every lane of a warp would land on the same bank, at 65 lane j
// reads bank (j + c) % 32.
constexpr int kAttnDTile = 64;

__global__ __launch_bounds__(kAttnGroup * 32, 12) void sliding_window_attention_warp_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, int q_heads, int kv_heads, int head_dim, int window,
    const int* __restrict__ position_of_row, const int* __restrict__ path, int max_len) {
    // Only the exponentials cross lanes, and only within a warp.
    __shared__ float sE[kAttnGroup][kWarpAttnMaxLen];
    // The group's K rows, shared by all four warps: they read identical rows,
    // and reading them from global once per warp had L1 at 97% of peak.
    __shared__ float sK[kWarpAttnMaxLen][kAttnDTile + 1];

    const int qi = blockIdx.y;
    const int kh = blockIdx.x;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int qh = kh * (q_heads / kv_heads) + warp;

    const int pos = position_of_row[qi];
    const int window_lo = pos - window + 1;
    const int lo = (window_lo > 0) ? window_lo : 0;
    const int len = pos - lo + 1;
    const int* __restrict__ keys = path + static_cast<std::size_t>(qi) * max_len + lo;
    const float scale = sqrtf(static_cast<float>(head_dim));
    const float* __restrict__ qv =
        q + static_cast<std::size_t>(qi) * q_heads * head_dim + static_cast<std::size_t>(qh) * head_dim;

    // (1) One key per lane, accumulating d = 0..head_dim-1 in the reference's
    // order. What changes is where the K element is read from.
    //
    // Reading it straight from global put the lanes of a warp on 32 *different*
    // rows, so one load asked L1 for 32 separate 32 B sectors; the four warps
    // then repeated all of it, since a GQA group's query heads share a kv head
    // and therefore share every K row they touch. Staging the tile cooperatively
    // makes the global side one contiguous run per row -- and the four warps
    // read it once between them instead of four times each.
    float score = 0.0f;
    for (int d0 = 0; d0 < head_dim; d0 += kAttnDTile) {
        __syncthreads();
        // 128 threads over len rows x kAttnDTile coordinates: consecutive
        // threads take consecutive coordinates of one row, so the global read
        // is a full line and the shared store hits 32 distinct banks.
        for (int t = static_cast<int>(threadIdx.x); t < len * kAttnDTile;
             t += static_cast<int>(blockDim.x)) {
            const int r = t / kAttnDTile, c = t - r * kAttnDTile;
            sK[r][c] = k[static_cast<std::size_t>(keys[r]) * kv_heads * head_dim +
                         static_cast<std::size_t>(kh) * head_dim + d0 + c];
        }
        __syncthreads();
        if (lane < len) {
            const float4* __restrict__ q4 = reinterpret_cast<const float4*>(qv + d0);
            // Measured worse unrolled: +0.5-0.7% end-to-end with the pragma
            // removed (ABBA, same srun allocation) -- unlike the GEMM's K-tile
            // and store-epilogue loops, where removing #pragma unroll cost
            // 7-10%. ptxas evidently schedules this 16-iteration loop better
            // left rolled.
            for (int c = 0; c < kAttnDTile; c += 4) {
                const float4 a = q4[c >> 2];
                score += a.x * sK[lane][c + 0];
                score += a.y * sK[lane][c + 1];
                score += a.z * sK[lane][c + 2];
                score += a.w * sK[lane][c + 3];
            }
        }
    }
    const float e = (lane < len) ? score / scale : 0.0f;

    // (2) Max, as a shuffle tree. Lanes past `len` carry -INFINITY, which is
    // fmaxf's identity, so a short row gives the same answer as a full one.
    float m = (lane < len) ? e : -INFINITY;
    // Same measured result as the dot-product loop above: rolled beats
    // unrolled here (+0.4-0.5%), the opposite of the GEMM's hot loops.
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

    // (3) Per-element, so whichever lane evaluates it produces the same bits.
    const float ex = (lane < len) ? static_cast<float>(exp(static_cast<double>(e - m))) : 0.0f;
    if (lane < len) sE[warp][lane] = ex;
    __syncwarp();

    // (4) Order-dependent, so still a sequential chain -- but every lane walks
    // its own copy instead of 127 threads parking on a barrier behind one.
    float denom = 0.0f;
    for (int idx = 0; idx < len; ++idx) denom += sE[warp][idx];

    // (5) Four output coordinates per lane, taken as one float4 so a warp's
    // V read is 512 B of a single line rather than four 128 B ones. Each
    // coordinate still gets its own accumulator walking idx = 0..len-1, and
    // the divide still happens per key -- hoisting a reciprocal out of the
    // loop would change the rounding.
    float* __restrict__ outv =
        out + static_cast<std::size_t>(qi) * q_heads * head_dim + static_cast<std::size_t>(qh) * head_dim;
    for (int d0 = lane * 4; d0 < head_dim; d0 += 128) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
        for (int idx = 0; idx < len; ++idx) {
            const float wgt = sE[warp][idx] / denom;
            const float4 vv = *reinterpret_cast<const float4*>(
                v + static_cast<std::size_t>(keys[idx]) * kv_heads * head_dim +
                static_cast<std::size_t>(kh) * head_dim + d0);
            ax += wgt * vv.x;
            ay += wgt * vv.y;
            az += wgt * vv.z;
            aw += wgt * vv.w;
        }
        *reinterpret_cast<float4*>(outv + d0) = make_float4(ax, ay, az, aw);
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

// Top-2 MoE routing, one thread per row. The reference's `select()` is not an
// argmax and `best_value` is not a running maximum: it accepts index e only
// when e beats the *last accepted* value by more than TIE_EPS, so the winner
// is a greedy staircase that depends on the scan order. Measured against this
// input's 498,656 real decisions, a stride-halving warp tree disagrees with it
// on 8.41% of them and an adjacent-pair tree on 0.21%; the sequential scan
// below agrees on all of them. So each thread runs both scans serially over
// its 16 registers -- there is no parallelism to lose, since the rows
// themselves supply 15,583-way work.
//
// The quantisation is the reference's expression unchanged. `fabsf` and
// `floorf` are exact, and nvcc's default -prec-div=true makes `/ QUANTUM` the
// same correctly-rounded divide x86 emits; an exhaustive sweep of all 2^32
// float bit patterns found zero host/device disagreements. Never rewrite the
// divide as a multiply by 1000.0f -- 1e-3f is 0.001000000047..., whose
// reciprocal is not 1000, and the substitution moves 1 value in 249,328.
__global__ void route_top2_kernel(const float* __restrict__ logits,
                                  int2* __restrict__ out, int rows) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows) return;
    const float* __restrict__ lg = logits + static_cast<std::size_t>(t) * apss26::NUM_EXPERTS;

    float s[apss26::NUM_EXPERTS];
#pragma unroll
    for (int e = 0; e < static_cast<int>(apss26::NUM_EXPERTS); ++e) {
        const float score = lg[e];
        const float rounded = floorf(fabsf(score) / apss26::ROUTER_SCORE_QUANTUM + 0.5f) *
                              apss26::ROUTER_SCORE_QUANTUM;
        s[e] = score < 0.0f ? -rounded : rounded;
    }

    int first = -1;
    float best = -INFINITY;
#pragma unroll
    for (int e = 0; e < static_cast<int>(apss26::NUM_EXPERTS); ++e)
        if (first < 0 || s[e] > best + apss26::ROUTER_TIE_EPS) { first = e; best = s[e]; }

    int second = -1;
    float best2 = -INFINITY;
#pragma unroll
    for (int e = 0; e < static_cast<int>(apss26::NUM_EXPERTS); ++e) {
        if (e == first) continue;
        if (second < 0 || s[e] > best2 + apss26::ROUTER_TIE_EPS) { second = e; best2 = s[e]; }
    }
    out[t] = make_int2(first, second);
}

// Grow-on-demand device storage for the pairs above.
struct RoutePairBuffer { int2* ptr = nullptr; std::size_t capacity = 0; };
RoutePairBuffer g_route_pairs;

// y[t,:] = 0.5*out[pos[2t],:] + 0.5*out[pos[2t+1],:]. Replaces a memset of
// d_y plus 16 serialized scatter_add_rows_kernel launches (one per expert,
// in ascending expert-id order) with one launch over rows*row_width.
//
// Bit-identical to that path even though this reads both of a token's expert
// outputs in one step rather than accumulating them in expert-id order,
// because the weight is always exactly 0.5 -- a power of two -- so
// 0.5f*out[...] never rounds (it only decrements the exponent). The old path
// computed round(0.5*a) via the first launch (exact, since a+0 needs no
// rounding either) then round(0.5*b + round(0.5*a)) via the second launch's
// FMA; since round(0.5*a) == 0.5*a exactly, that is just
// round(0.5*a_exact + 0.5*b_exact) -- the single correctly-rounded value
// this kernel's own `0.5f*a + 0.5f*b` (fused into an FMA or not) computes,
// regardless of which of a token's two experts pos[] lists first.
__global__ void scatter_pairs_kernel(const float* __restrict__ out,
                                     const int* __restrict__ pos,
                                     float* __restrict__ y,
                                     std::size_t rows, std::size_t row_width) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = rows * row_width;
    if (idx >= total) return;
    const std::size_t t = idx / row_width;
    const std::size_t j = idx % row_width;
    const std::size_t p0 = static_cast<std::size_t>(pos[2 * t]);
    const std::size_t p1 = static_cast<std::size_t>(pos[2 * t + 1]);
    y[idx] = 0.5f * out[p0 * row_width + j] + 0.5f * out[p1 * row_width + j];
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
        // The *tile* is settled at 128x128: 128x256 and 256x128 (512 threads)
        // raise arithmetic intensity 32 -> 42.7, past the 38 flop/byte where
        // this GPU turns compute-bound, and do cut traffic 1.34x as predicted,
        // and still lost ~7% -- one 512-thread block per SM means a
        // __syncthreads() stalls all 16 warps where two blocks would cover for
        // each other (bandwidth fell 692 -> 489 GB/s).
        //
        // The *micro-tile* was not settled, and 8x8 was the wrong choice.
        // NT is derived: NT = BM*BN/(TM*TN), so every earlier attempt to move
        // it moved the tile instead, and 16x8 was never tried. It matters
        // because 8x8 sits exactly on the sm_86 shared/FFMA wall -- per k step
        // 4 LDS.128 is 2048 B / 128 B-per-clk = 16 clk against 64 FFMA / 4 =
        // 16 clk, 1:1 with no slack. At 16x8 it is 6 LDS.128 (24 clk) against
        // 128 FFMA (32 clk): 1:1.33, and the wall moves. Shared floats per
        // FFMA go 4.0 -> 5.33.
        //
        // The price is occupancy: NT drops to 128, so 8 warps/SM = 16.7%,
        // half of what already looked low. Measured 2026-08-26 anyway:
        // +7.9% end-to-end, bit-identical. This is the third and strongest
        // confirmation that occupancy is the wrong knob for this kernel --
        // forcing it *up* to 50% costs 20% (see kernels.md).
        //
        // MINB must be 1 here. At 248 registers a hint of 2 makes ptxas cut
        // registers and spill; left free it allocates 248 and 248*128 = 31744
        // still fits two blocks in the 32768-register half-SM anyway. That
        // leaves only 8 registers/thread of headroom -- if a compiler update
        // pushes past it the SM drops to 1 block and the kernel halves. Check
        // `nvcc -Xptxas=-v | grep Li128ELi128ELi16ELi8` after touching this.
        const dim3 block(128 / 8, 128 / 16);         // 16x8 = 128
        const dim3 grid(static_cast<unsigned>((n + 127) / 128),
                        static_cast<unsigned>((m + 127) / 128));
        if (vec) matmul_transposed_blocked_kernel<128, 128, 16, 8, true, 128, 1><<<grid, block>>>(
                     d_a, d_b, d_c, (int)m, (int)k, (int)n, nullptr, 0, d_bias);
        else     matmul_transposed_blocked_kernel<128, 128, 16, 8, false, 128, 1><<<grid, block>>>(
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
        const dim3 block(128 / 8, 128 / 16);
        const dim3 grid(static_cast<unsigned>((n + 127) / 128), static_cast<unsigned>(num_tiles));
        if (vec) matmul_transposed_blocked_kernel<128, 128, 16, 8, true, 128, 1><<<grid, block>>>(
                     d_a, d_b, d_c, 0, ki, ni, d_tiles, stride, nullptr);
        else     matmul_transposed_blocked_kernel<128, 128, 16, 8, false, 128, 1><<<grid, block>>>(
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
    // The skeleton's Tensor form has no trie behind it: every row is its own
    // sequence position, so the path is the identity run 0..seq_len-1.
    std::vector<int> pos(seq_len), path(seq_len * seq_len);
    for (std::size_t r = 0; r < seq_len; ++r) {
        pos[r] = static_cast<int>(r);
        for (std::size_t j = 0; j <= r; ++j) path[r * seq_len + j] = static_cast<int>(j);
    }
    const device::RowMap rows = device::upload_row_map(pos.data(), path.data(), seq_len, seq_len);
    // shared_e only ever holds `len` entries, and len <= min(window, max_len)
    // -- the trie caps a row's key list at max_len (32 for this input) while
    // window is 2047. Sizing by window alone asked for 8188 B to use 128 B,
    // and ncu put Block Limit Shared Mem at 10 against Block Limit Warps 12:
    // the over-allocation, not the kernel, was capping occupancy.
    const std::size_t shared_bytes =
        (window < static_cast<std::size_t>(rows.max_len) ? window
                                                         : static_cast<std::size_t>(rows.max_len)) * sizeof(float);
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

void silu_mul_fused(const float* d_gateup, float* d_out, std::size_t rows, std::size_t inter) {
    const std::size_t total = rows * inter;
    if (total == 0) return;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    silu_mul_fused_kernel<<<blocks, threads>>>(d_gateup, d_out, rows, static_cast<int>(inter));
    cuda_check(cudaGetLastError(), "device::silu_mul_fused kernel");
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

namespace {
struct IntPosBuffer { int* ptr = nullptr; std::size_t capacity = 0; };
IntPosBuffer g_scatter_pos;
}  // namespace

void scatter_pairs(const float* d_out, const int* host_pos, float* d_y,
                   std::size_t rows, std::size_t row_width) {
    if (rows == 0) return;
    const std::size_t n = 2 * rows;
    if (g_scatter_pos.capacity < n) {
        if (g_scatter_pos.ptr) cudaFree(g_scatter_pos.ptr);
        cuda_check(cudaMalloc(&g_scatter_pos.ptr, n * sizeof(int)), "scatter_pos cudaMalloc");
        g_scatter_pos.capacity = n;
    }
    cuda_check(cudaMemcpy(g_scatter_pos.ptr, host_pos, n * sizeof(int), cudaMemcpyHostToDevice),
              "scatter_pos H2D");
    const std::size_t total = rows * row_width;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    scatter_pairs_kernel<<<blocks, threads>>>(d_out, g_scatter_pos.ptr, d_y, rows, row_width);
    cuda_check(cudaGetLastError(), "scatter_pairs kernel");
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
    // A row's key list is capped by both the window and the trie's max_len.
    // When that cap fits in a warp the barrier-free kernel above applies; the
    // general one stays for any shape it does not cover (the Tensor-based
    // entry point below builds an identity path, where max_len is seq_len).
    const std::size_t key_cap =
        (window < static_cast<std::size_t>(rows.max_len) ? window
                                                         : static_cast<std::size_t>(rows.max_len));
    if (key_cap <= static_cast<std::size_t>(kWarpAttnMaxLen) &&
        kv_heads * kAttnGroup == q_heads && head_dim % kAttnDTile == 0) {
        const dim3 grid(static_cast<unsigned>(kv_heads), static_cast<unsigned>(seq_len));
        sliding_window_attention_warp_kernel<<<grid, kAttnGroup * 32>>>(
            d_q, d_k, d_v, d_out, static_cast<int>(q_heads),
            static_cast<int>(kv_heads), static_cast<int>(head_dim), static_cast<int>(window),
            rows.position, rows.path, rows.max_len);
        cuda_check(cudaGetLastError(), "device::sliding_window_attention warp kernel");
        return;
    }

    const dim3 grid(static_cast<unsigned>(q_heads), static_cast<unsigned>(seq_len));
    // shared_e only ever holds `len` entries, so size it by that cap rather
    // than by the window: this asked for 8188 B to use 128 B, and ncu put
    // Block Limit Shared Mem at 10 against Block Limit Warps 12 -- the
    // over-allocation, not the kernel, was capping occupancy.
    const std::size_t shared_bytes = key_cap * sizeof(float);
    sliding_window_attention_kernel<<<grid, kAttnThreads, shared_bytes>>>(
        d_q, d_k, d_v, d_out, static_cast<int>(seq_len), static_cast<int>(q_heads),
        static_cast<int>(kv_heads), static_cast<int>(head_dim), static_cast<int>(window),
        rows.position, rows.path, rows.max_len);
    cuda_check(cudaGetLastError(), "device::sliding_window_attention kernel");
}

// The gate scores stay on the device: the 997 KB download they used to need
// was a blocking sync 32 times over, and nsys measured 57 ms of GPU idle
// behind it. What comes back instead is 2 ints per row.
void route_top2(const float* d_router, int* host_pairs, std::size_t rows) {
    if (rows == 0) return;
    if (g_route_pairs.capacity < rows) {
        if (g_route_pairs.ptr) cudaFree(g_route_pairs.ptr);
        check(cudaMalloc(&g_route_pairs.ptr, rows * sizeof(int2)), "route_top2 cudaMalloc");
        g_route_pairs.capacity = rows;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    route_top2_kernel<<<blocks, threads>>>(d_router, g_route_pairs.ptr, static_cast<int>(rows));
    check(cudaGetLastError(), "route_top2 kernel");
    check(cudaMemcpy(host_pairs, g_route_pairs.ptr, rows * sizeof(int2), cudaMemcpyDeviceToHost),
          "route_top2 pairs D2H");
}

}  // namespace device
}  // namespace tensor_ops
