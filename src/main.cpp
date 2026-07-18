// main.cpp — run the CPU reference and the GPU primitive, check equivalence,
// and report timing HONESTLY: the GPU's one-time CUDA context-init cost is
// separated from warm steady-state compute (the case that matters when STA is
// called repeatedly inside a placement / optimization loop). No device -> the
// GPU path falls back to the CPU reference and says so.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sta.h"

using Clock = std::chrono::steady_clock;

static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    const int levels = argc > 1 ? std::atoi(argv[1]) : 400;
    const int width = argc > 2 ? std::atoi(argv[2]) : 4000;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;

    egg::TimingGraph g = egg::generateLayeredDag(levels, width, 4, 0x9E3779B97F4A7C15ULL);
    std::printf("timing graph: nodes=%d  levels=%d  arcs=%d\n", g.numNodes,
                g.numLevels, g.numArcs());

    // CPU: warm the caches, then take the min of `reps` runs.
    egg::TimingResult cpu = egg::staCpu(g);
    double cpuMs = std::numeric_limits<double>::infinity();
    for (int i = 0; i < reps; ++i) {
        auto t0 = Clock::now();
        cpu = egg::staCpu(g);
        cpuMs = std::min(cpuMs, msSince(t0));
    }

    // Fair all-core CPU baseline: same level decomposition, every processor OpenMP
    // can see (set explicitly so the count is deterministic, not at the mercy of an
    // ambient OMP_NUM_THREADS or an inherited CPU-affinity mask).
#ifdef _OPENMP
    const int cpuThreads = omp_get_num_procs();
#else
    const int cpuThreads = 1;
#endif
    egg::TimingResult cpuMt = egg::staCpuParallel(g, cpuThreads);
    double cpuMtMs = std::numeric_limits<double>::infinity();
    for (int i = 0; i < reps; ++i) {
        auto t0 = Clock::now();
        cpuMt = egg::staCpuParallel(g, cpuThreads);
        cpuMtMs = std::min(cpuMtMs, msSince(t0));
    }
    const double cpuMtDiff = egg::maxAbsDiff(cpu, cpuMt);  // must be bit-exact

    // GPU: the FIRST call pays one-time CUDA context init (+ any JIT). Time it
    // separately and label it honestly — it is not the steady-state compute cost.
    bool ranGpu = false;
    auto tc = Clock::now();
    egg::TimingResult gpu = egg::staGpu(g, &ranGpu);
    const double coldMs = msSince(tc);

    std::printf("critical-path length (period): %.3f\n", cpu.period);
    std::printf("CPU STA  1-core  (min of %d):    %8.2f ms\n", reps, cpuMs);
    std::printf("CPU STA %2d-core  (min of %d):    %8.2f ms   (%.2fx vs 1-core, oracle %s)\n",
                cpuThreads, reps, cpuMtMs, cpuMtMs > 0 ? cpuMs / cpuMtMs : 0.0,
                cpuMtDiff == 0.0 ? "exact" : "MISMATCH");

    if (!ranGpu) {
        std::printf("GPU STA: no CUDA device — fell back to the CPU reference.\n");
        return 0;
    }

    // Warm steady-state: context is now live; take the min of `reps` runs.
    double gpuMs = std::numeric_limits<double>::infinity();
    for (int i = 0; i < reps; ++i) {
        auto t0 = Clock::now();
        gpu = egg::staGpu(g, &ranGpu);
        gpuMs = std::min(gpuMs, msSince(t0));
    }
    const double d = egg::maxAbsDiff(cpu, gpu);
    const bool match = d <= 1e-2;

    std::printf("GPU STA cold (incl. 1-time init): %8.2f ms\n", coldMs);
    std::printf("GPU STA warm (min of %d):          %8.2f ms   (%.2fx vs 1-core, %.2fx vs %d-core)\n",
                reps, gpuMs, gpuMs > 0 ? cpuMs / gpuMs : 0.0,
                gpuMs > 0 ? cpuMtMs / gpuMs : 0.0, cpuThreads);

    // Improvement iteration 2: capture the level-sweep once, replay it. This is the
    // incremental-STA pattern — the capture is a one-time build cost, and each
    // staGpuPlanRun replays with a single cudaGraphLaunch (no per-level launches).
    auto tb = Clock::now();
    egg::StaGpuPlan* plan = egg::staGpuPlanCreate(g);
    const double buildMs = msSince(tb);
    double graphMs = std::numeric_limits<double>::infinity();
    double dg = 0.0;
    if (plan) {
        egg::TimingResult gp = egg::staGpuPlanRun(plan);  // warm-up replay
        for (int i = 0; i < reps; ++i) {
            auto t0 = Clock::now();
            gp = egg::staGpuPlanRun(plan);
            graphMs = std::min(graphMs, msSince(t0));
        }
        dg = egg::maxAbsDiff(cpu, gp);
        egg::staGpuPlanDestroy(plan);
        std::printf("GPU STA graph-replay (min of %d): %8.2f ms   (%.2fx vs 1-core, "
                    "%.2fx vs %d-core, %.2fx vs per-launch; capture %.2f ms)\n",
                    reps, graphMs, graphMs > 0 ? cpuMs / graphMs : 0.0,
                    graphMs > 0 ? cpuMtMs / graphMs : 0.0, cpuThreads,
                    graphMs > 0 ? gpuMs / graphMs : 0.0, buildMs);
    }

    const double dAll = std::max(std::max(d, dg), cpuMtDiff);
    std::printf("oracle: max|arrival/required diff| = %.3e  ->  %s\n", dAll,
                (dAll <= 1e-2) ? "MATCH" : "MISMATCH");
    return (dAll <= 1e-2) ? 0 : 1;
}
