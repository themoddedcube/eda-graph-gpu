// sta_gpu.cu — level-parallel STA primitive on the GPU.
//
// One kernel launch per topological level; every node in the level is updated by
// one thread in parallel (all its fanin already resolved at lower levels). This
// is the primitive no vendor library provides — the reason eda-graph-gpu exists.
//
// Improvement iteration 1 (see docs/research/sta-optimization-roadmap.md, change #1):
// the critical-path period is reduced ON DEVICE with cub::DeviceReduce::Max over the
// PO-masked arrival array, and slack is FUSED into the backward kernel. That removes
// the old mid-pipeline host round-trip (sync + arrival D2H + host period loop) and
// the separate host slack loop — only a single sync + one D2H of the results remain.
// The math is identical to the CPU reference, so the maxAbsDiff oracle still holds by
// construction; the win is launch/round-trip overhead, to be MEASURED on a GPU host.
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <vector>

#include "sta.h"

namespace egg {

namespace {

#define EGG_CUDA_OK(call)                                       \
    do {                                                        \
        cudaError_t _e = (call);                                \
        if (_e != cudaSuccess) return false;                    \
    } while (0)

// Forward: arrival[v] = max over fanin (arrival[u] + delay); PI -> 0.
__global__ void arrivalKernel(int lstart, int lend, const int* levelNodes,
                              const int* finStart, const int* finFrom,
                              const float* finDelay, float* arrival) {
    const int idx = lstart + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lend) return;
    const int v = levelNodes[idx];
    const int s = finStart[v], e = finStart[v + 1];
    if (s == e) { arrival[v] = 0.0f; return; }
    float a = -3.402823e38f;  // -FLT_MAX
    for (int p = s; p < e; ++p) a = fmaxf(a, arrival[finFrom[p]] + finDelay[p]);
    arrival[v] = a;
}

// Mask for the on-device period reduction: a primary output (no fanout) keeps its
// arrival; everything else maps to -inf so a plain max reduction yields
// max-over-POs — definitionally the same period the CPU reference computes.
__global__ void poMaskKernel(int n, const int* foutStart, const float* arrival,
                             float* masked) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const bool isPO = (foutStart[v] == foutStart[v + 1]);
    masked[v] = isPO ? arrival[v] : -3.402823e38f;
}

// Backward: required[v] = min over fanout (required[w] - delay); PO -> period.
// Slack is fused in (required - arrival) so no separate pass is needed. `period`
// is a device pointer (produced by the on-device reduction), not a host value.
__global__ void requiredKernel(int lstart, int lend, const int* levelNodes,
                               const int* foutStart, const int* foutTo,
                               const float* foutDelay, const float* period,
                               const float* arrival, float* required, float* slack) {
    const int idx = lstart + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lend) return;
    const int v = levelNodes[idx];
    const int s = foutStart[v], e = foutStart[v + 1];
    float r;
    if (s == e) {
        r = *period;
    } else {
        r = 3.402823e38f;  // +FLT_MAX
        for (int p = s; p < e; ++p) r = fminf(r, required[foutTo[p]] - foutDelay[p]);
    }
    required[v] = r;
    slack[v] = r - arrival[v];
}

template <typename T>
bool upload(const std::vector<T>& h, T*& d) {
    EGG_CUDA_OK(cudaMalloc(&d, h.size() * sizeof(T)));
    EGG_CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return true;
}

bool runGpu(const TimingGraph& g, TimingResult& out) {
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) return false;

    const int n = g.numNodes;
    int *dLevelNodes = nullptr, *dFinStart = nullptr, *dFinFrom = nullptr;
    int *dFoutStart = nullptr, *dFoutTo = nullptr;
    float *dFinDelay = nullptr, *dFoutDelay = nullptr, *dArrival = nullptr,
          *dRequired = nullptr, *dSlack = nullptr, *dMasked = nullptr, *dPeriod = nullptr;
    void* dTemp = nullptr;
    if (!upload(g.levelNodes, dLevelNodes) || !upload(g.finStart, dFinStart) ||
        !upload(g.finFrom, dFinFrom) || !upload(g.finDelay, dFinDelay) ||
        !upload(g.foutStart, dFoutStart) || !upload(g.foutTo, dFoutTo) ||
        !upload(g.foutDelay, dFoutDelay))
        return false;
    EGG_CUDA_OK(cudaMalloc(&dArrival, n * sizeof(float)));
    EGG_CUDA_OK(cudaMalloc(&dRequired, n * sizeof(float)));
    EGG_CUDA_OK(cudaMalloc(&dSlack, n * sizeof(float)));
    EGG_CUDA_OK(cudaMalloc(&dMasked, n * sizeof(float)));
    EGG_CUDA_OK(cudaMalloc(&dPeriod, sizeof(float)));

    const int TPB = 256;
    // Forward sweep: level by level (serial across levels, parallel within). All on
    // the default stream, so no host sync is needed between levels or between sweeps.
    for (int L = 0; L < g.numLevels; ++L) {
        const int lo = g.levelStart[L], hi = g.levelStart[L + 1];
        const int count = hi - lo;
        if (count <= 0) continue;
        arrivalKernel<<<(count + TPB - 1) / TPB, TPB>>>(
            lo, hi, dLevelNodes, dFinStart, dFinFrom, dFinDelay, dArrival);
    }

    // Period on device: mask non-POs to -inf, then cub max-reduce -> dPeriod.
    poMaskKernel<<<(n + TPB - 1) / TPB, TPB>>>(n, dFoutStart, dArrival, dMasked);
    size_t tempBytes = 0;
    EGG_CUDA_OK(cub::DeviceReduce::Max(nullptr, tempBytes, dMasked, dPeriod, n));
    EGG_CUDA_OK(cudaMalloc(&dTemp, tempBytes));
    EGG_CUDA_OK(cub::DeviceReduce::Max(dTemp, tempBytes, dMasked, dPeriod, n));

    // Backward sweep from the last level down; reads dPeriod, writes required + slack.
    for (int L = g.numLevels - 1; L >= 0; --L) {
        const int lo = g.levelStart[L], hi = g.levelStart[L + 1];
        const int count = hi - lo;
        if (count <= 0) continue;
        requiredKernel<<<(count + TPB - 1) / TPB, TPB>>>(
            lo, hi, dLevelNodes, dFoutStart, dFoutTo, dFoutDelay, dPeriod, dArrival,
            dRequired, dSlack);
    }

    EGG_CUDA_OK(cudaDeviceSynchronize());  // single sync for the whole pipeline

    out.arrival.resize(n);
    out.required.resize(n);
    out.slack.resize(n);
    EGG_CUDA_OK(cudaMemcpy(out.arrival.data(), dArrival, n * sizeof(float),
                           cudaMemcpyDeviceToHost));
    EGG_CUDA_OK(cudaMemcpy(out.required.data(), dRequired, n * sizeof(float),
                           cudaMemcpyDeviceToHost));
    EGG_CUDA_OK(cudaMemcpy(out.slack.data(), dSlack, n * sizeof(float),
                           cudaMemcpyDeviceToHost));
    float period = 0.0f;
    EGG_CUDA_OK(cudaMemcpy(&period, dPeriod, sizeof(float), cudaMemcpyDeviceToHost));
    out.period = period;

    cudaFree(dLevelNodes); cudaFree(dFinStart); cudaFree(dFinFrom);
    cudaFree(dFinDelay); cudaFree(dFoutStart); cudaFree(dFoutTo);
    cudaFree(dFoutDelay); cudaFree(dArrival); cudaFree(dRequired);
    cudaFree(dSlack); cudaFree(dMasked); cudaFree(dPeriod); cudaFree(dTemp);
    return true;
}

}  // namespace

TimingResult staGpu(const TimingGraph& g, bool* ranGpu) {
    TimingResult out;
    if (runGpu(g, out)) {
        if (ranGpu) *ranGpu = true;
        return out;
    }
    // Honest fallback: no device (or a CUDA error) -> run the CPU reference, so
    // the same binary is correct everywhere and never fakes a GPU result.
    if (ranGpu) *ranGpu = false;
    return staCpu(g);
}

}  // namespace egg
