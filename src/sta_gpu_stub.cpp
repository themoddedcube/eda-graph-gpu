// sta_gpu_stub.cpp — CPU-only build stub for hosts without nvcc/CUDA.
// Linked instead of sta_gpu.cu by `make cpu`; keeps the same honest contract.
#include "sta.h"
namespace egg {
TimingResult staGpu(const TimingGraph& g, bool* ranGpu) {
    if (ranGpu) *ranGpu = false;
    return staCpu(g);
}

// No CUDA here: no plan can be built, so callers fall back to staCpu.
StaGpuPlan* staGpuPlanCreate(const TimingGraph&) { return nullptr; }
bool staGpuPlanUpdateDelays(StaGpuPlan*, const std::vector<float>&,
                            const std::vector<float>&) {
    return false;
}
TimingResult staGpuPlanRun(StaGpuPlan*) { return TimingResult(); }
void staGpuPlanDestroy(StaGpuPlan*) {}
}  // namespace egg
