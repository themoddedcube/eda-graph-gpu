// sta_gpu.cu — level-parallel STA primitive on the GPU.
//
// One kernel launch per topological level; every node in the level is updated by
// one thread in parallel (all its fanin already resolved at lower levels). This
// is the primitive no vendor library provides — the reason eda-graph-gpu exists.
#include <cuda_runtime.h>

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

// Backward: required[v] = min over fanout (required[w] - delay); PO -> period.
__global__ void requiredKernel(int lstart, int lend, const int* levelNodes,
                               const int* foutStart, const int* foutTo,
                               const float* foutDelay, float period,
                               float* required) {
    const int idx = lstart + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lend) return;
    const int v = levelNodes[idx];
    const int s = foutStart[v], e = foutStart[v + 1];
    if (s == e) { required[v] = period; return; }
    float r = 3.402823e38f;  // +FLT_MAX
    for (int p = s; p < e; ++p) r = fminf(r, required[foutTo[p]] - foutDelay[p]);
    required[v] = r;
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

    int *dLevelNodes = nullptr, *dFinStart = nullptr, *dFinFrom = nullptr;
    int *dFoutStart = nullptr, *dFoutTo = nullptr;
    float *dFinDelay = nullptr, *dFoutDelay = nullptr, *dArrival = nullptr,
          *dRequired = nullptr;
    if (!upload(g.levelNodes, dLevelNodes) || !upload(g.finStart, dFinStart) ||
        !upload(g.finFrom, dFinFrom) || !upload(g.finDelay, dFinDelay) ||
        !upload(g.foutStart, dFoutStart) || !upload(g.foutTo, dFoutTo) ||
        !upload(g.foutDelay, dFoutDelay))
        return false;
    EGG_CUDA_OK(cudaMalloc(&dArrival, g.numNodes * sizeof(float)));
    EGG_CUDA_OK(cudaMalloc(&dRequired, g.numNodes * sizeof(float)));

    const int TPB = 256;
    // Forward sweep: level by level (serial across levels, parallel within).
    for (int L = 0; L < g.numLevels; ++L) {
        const int lo = g.levelStart[L], hi = g.levelStart[L + 1];
        const int count = hi - lo;
        if (count <= 0) continue;
        arrivalKernel<<<(count + TPB - 1) / TPB, TPB>>>(
            lo, hi, dLevelNodes, dFinStart, dFinFrom, dFinDelay, dArrival);
    }
    EGG_CUDA_OK(cudaDeviceSynchronize());

    out.arrival.resize(g.numNodes);
    EGG_CUDA_OK(cudaMemcpy(out.arrival.data(), dArrival, g.numNodes * sizeof(float),
                           cudaMemcpyDeviceToHost));

    float period = 0.0f;
    for (int v = 0; v < g.numNodes; ++v)
        if (g.isPrimaryOutput(v)) period = fmaxf(period, out.arrival[v]);
    out.period = period;

    // Backward sweep from the last level down.
    for (int L = g.numLevels - 1; L >= 0; --L) {
        const int lo = g.levelStart[L], hi = g.levelStart[L + 1];
        const int count = hi - lo;
        if (count <= 0) continue;
        requiredKernel<<<(count + TPB - 1) / TPB, TPB>>>(
            lo, hi, dLevelNodes, dFoutStart, dFoutTo, dFoutDelay, period, dRequired);
    }
    EGG_CUDA_OK(cudaDeviceSynchronize());

    out.required.resize(g.numNodes);
    EGG_CUDA_OK(cudaMemcpy(out.required.data(), dRequired, g.numNodes * sizeof(float),
                           cudaMemcpyDeviceToHost));

    out.slack.resize(g.numNodes);
    for (int v = 0; v < g.numNodes; ++v) out.slack[v] = out.required[v] - out.arrival[v];

    cudaFree(dLevelNodes); cudaFree(dFinStart); cudaFree(dFinFrom);
    cudaFree(dFinDelay); cudaFree(dFoutStart); cudaFree(dFoutTo);
    cudaFree(dFoutDelay); cudaFree(dArrival); cudaFree(dRequired);
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
