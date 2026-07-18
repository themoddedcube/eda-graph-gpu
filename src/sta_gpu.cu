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

// --- Persistent GPU plan: capture the level-sweep once, replay per run ---

struct StaGpuPlan {
    int n = 0, numLevels = 0;
    int *dLevelNodes = nullptr, *dFinStart = nullptr, *dFinFrom = nullptr;
    int *dFoutStart = nullptr, *dFoutTo = nullptr;
    float *dFinDelay = nullptr, *dFoutDelay = nullptr, *dArrival = nullptr;
    float *dRequired = nullptr, *dSlack = nullptr, *dMasked = nullptr, *dPeriod = nullptr;
    void* dTemp = nullptr;
    size_t tempBytes = 0;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
};

void staGpuPlanDestroy(StaGpuPlan* p) {
    if (!p) return;
    if (p->exec) cudaGraphExecDestroy(p->exec);
    if (p->graph) cudaGraphDestroy(p->graph);
    if (p->stream) cudaStreamDestroy(p->stream);
    cudaFree(p->dLevelNodes); cudaFree(p->dFinStart); cudaFree(p->dFinFrom);
    cudaFree(p->dFinDelay); cudaFree(p->dFoutStart); cudaFree(p->dFoutTo);
    cudaFree(p->dFoutDelay); cudaFree(p->dArrival); cudaFree(p->dRequired);
    cudaFree(p->dSlack); cudaFree(p->dMasked); cudaFree(p->dPeriod); cudaFree(p->dTemp);
    delete p;
}

StaGpuPlan* staGpuPlanCreate(const TimingGraph& g) {
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) return nullptr;
    StaGpuPlan* p = new StaGpuPlan();
    p->n = g.numNodes;
    p->numLevels = g.numLevels;
    const int n = p->n;

    if (!upload(g.levelNodes, p->dLevelNodes) || !upload(g.finStart, p->dFinStart) ||
        !upload(g.finFrom, p->dFinFrom) || !upload(g.finDelay, p->dFinDelay) ||
        !upload(g.foutStart, p->dFoutStart) || !upload(g.foutTo, p->dFoutTo) ||
        !upload(g.foutDelay, p->dFoutDelay)) {
        staGpuPlanDestroy(p);
        return nullptr;
    }
    if (cudaMalloc(&p->dArrival, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&p->dRequired, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&p->dSlack, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&p->dMasked, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&p->dPeriod, sizeof(float)) != cudaSuccess) {
        staGpuPlanDestroy(p);
        return nullptr;
    }
    cub::DeviceReduce::Max(nullptr, p->tempBytes, p->dMasked, p->dPeriod, n);
    if (cudaMalloc(&p->dTemp, p->tempBytes) != cudaSuccess ||
        cudaStreamCreate(&p->stream) != cudaSuccess) {
        staGpuPlanDestroy(p);
        return nullptr;
    }

    // Capture the whole forward + reduce + backward chain into one CUDA graph.
    const int TPB = 256;
    if (cudaStreamBeginCapture(p->stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        staGpuPlanDestroy(p);
        return nullptr;
    }
    for (int L = 0; L < g.numLevels; ++L) {
        const int lo = g.levelStart[L], hi = g.levelStart[L + 1], count = hi - lo;
        if (count <= 0) continue;
        arrivalKernel<<<(count + TPB - 1) / TPB, TPB, 0, p->stream>>>(
            lo, hi, p->dLevelNodes, p->dFinStart, p->dFinFrom, p->dFinDelay, p->dArrival);
    }
    poMaskKernel<<<(n + TPB - 1) / TPB, TPB, 0, p->stream>>>(n, p->dFoutStart,
                                                            p->dArrival, p->dMasked);
    cub::DeviceReduce::Max(p->dTemp, p->tempBytes, p->dMasked, p->dPeriod, n, p->stream);
    for (int L = g.numLevels - 1; L >= 0; --L) {
        const int lo = g.levelStart[L], hi = g.levelStart[L + 1], count = hi - lo;
        if (count <= 0) continue;
        requiredKernel<<<(count + TPB - 1) / TPB, TPB, 0, p->stream>>>(
            lo, hi, p->dLevelNodes, p->dFoutStart, p->dFoutTo, p->dFoutDelay, p->dPeriod,
            p->dArrival, p->dRequired, p->dSlack);
    }
    cudaGraph_t graph = nullptr;
    if (cudaStreamEndCapture(p->stream, &graph) != cudaSuccess || graph == nullptr) {
        staGpuPlanDestroy(p);
        return nullptr;
    }
    p->graph = graph;
    if (cudaGraphInstantiateWithFlags(&p->exec, graph, 0) != cudaSuccess) {
        staGpuPlanDestroy(p);
        return nullptr;
    }
    return p;
}

TimingResult staGpuPlanRun(StaGpuPlan* p) {
    TimingResult out;
    if (!p || !p->exec) return out;
    const int n = p->n;
    if (cudaGraphLaunch(p->exec, p->stream) != cudaSuccess) return out;
    if (cudaStreamSynchronize(p->stream) != cudaSuccess) return out;
    out.arrival.resize(n);
    out.required.resize(n);
    out.slack.resize(n);
    cudaMemcpy(out.arrival.data(), p->dArrival, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.required.data(), p->dRequired, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.slack.data(), p->dSlack, n * sizeof(float), cudaMemcpyDeviceToHost);
    float period = 0.0f;
    cudaMemcpy(&period, p->dPeriod, sizeof(float), cudaMemcpyDeviceToHost);
    out.period = period;
    return out;
}

}  // namespace egg
