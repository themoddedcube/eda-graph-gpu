// main.cpp — run the CPU reference and the GPU primitive, check equivalence,
// and report the measured speedup (honest: falls back + says so with no device).
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "sta.h"

int main(int argc, char** argv) {
    const int levels = argc > 1 ? std::atoi(argv[1]) : 400;
    const int width = argc > 2 ? std::atoi(argv[2]) : 4000;

    egg::TimingGraph g = egg::generateLayeredDag(levels, width, 4, 0x9E3779B97F4A7C15ULL);
    std::printf("timing graph: nodes=%d  levels=%d  arcs=%d\n", g.numNodes,
                g.numLevels, g.numArcs());

    auto t0 = std::chrono::steady_clock::now();
    egg::TimingResult cpu = egg::staCpu(g);
    auto t1 = std::chrono::steady_clock::now();
    const double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ranGpu = false;
    auto t2 = std::chrono::steady_clock::now();
    egg::TimingResult gpu = egg::staGpu(g, &ranGpu);
    auto t3 = std::chrono::steady_clock::now();
    const double gpuMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::printf("critical-path length (period): %.3f\n", cpu.period);
    std::printf("CPU STA: %8.2f ms\n", cpuMs);

    if (ranGpu) {
        const double d = egg::maxAbsDiff(cpu, gpu);
        const bool match = d <= 1e-2;
        std::printf("GPU STA: %8.2f ms  (%.1fx)\n", gpuMs,
                    gpuMs > 0 ? cpuMs / gpuMs : 0.0);
        std::printf("oracle: max|arrival/required diff| = %.3e  ->  %s\n", d,
                    match ? "MATCH" : "MISMATCH");
        return match ? 0 : 1;
    }
    std::printf("GPU STA: no CUDA device — fell back to the CPU reference.\n");
    std::printf("(.cu compiles to an sm_XX fatbin here; run on a GPU host for the "
                "oracle check + measured speedup.)\n");
    return 0;
}
