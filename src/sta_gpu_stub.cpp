// sta_gpu_stub.cpp — CPU-only build stub for hosts without nvcc/CUDA.
// Linked instead of sta_gpu.cu by `make cpu`; keeps the same honest contract.
#include "sta.h"
namespace egg {
TimingResult staGpu(const TimingGraph& g, bool* ranGpu) {
    if (ranGpu) *ranGpu = false;
    return staCpu(g);
}
}  // namespace egg
