// sta_cpu_mt.cpp — multi-threaded CPU reference (OpenMP).
//
// A FAIR CPU baseline: real STA engines (OpenTimer, etc.) use every core, so
// comparing the GPU only against one CPU thread overstates the win. This uses the
// SAME decomposition as the GPU primitive — nodes within a topological level are
// independent and updated in parallel; levels run serially. The per-node float ops
// are identical to staCpu, and the only reduction (period = max over POs) is
// order-independent, so the result is bit-identical to staCpu (asserted in tests).
//
// Without OpenMP (`-fopenmp` absent) the pragmas are ignored and this is just a
// correct serial pass — so the file always compiles and stays honest.
#include <algorithm>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sta.h"

namespace egg {

TimingResult staCpuParallel(const TimingGraph& g, int numThreads) {
#ifdef _OPENMP
    // Default (numThreads<=0) to every processor OpenMP can see — a fair baseline
    // uses the whole CPU, and this is deterministic regardless of a stale ambient
    // OMP_NUM_THREADS. A caller can still pin a specific count.
    const int t = (numThreads > 0) ? numThreads : omp_get_num_procs();
    omp_set_num_threads(t);
#else
    (void)numThreads;
#endif
    TimingResult r;
    r.arrival.assign(g.numNodes, 0.0f);

    // Forward: parallel within each level (fanin already resolved at lower levels).
    for (int L = 0; L < g.numLevels; ++L) {
        const int lo = g.levelStart[L], hi = g.levelStart[L + 1];
#pragma omp parallel for schedule(static)
        for (int idx = lo; idx < hi; ++idx) {
            const int v = g.levelNodes[idx];
            const int s = g.finStart[v], e = g.finStart[v + 1];
            if (s == e) { r.arrival[v] = 0.0f; continue; }
            float a = -std::numeric_limits<float>::infinity();
            for (int p = s; p < e; ++p)
                a = std::max(a, r.arrival[g.finFrom[p]] + g.finDelay[p]);
            r.arrival[v] = a;
        }
    }

    // Period = max arrival over primary outputs (order-independent reduction).
    float period = 0.0f;
#pragma omp parallel for reduction(max : period) schedule(static)
    for (int v = 0; v < g.numNodes; ++v)
        if (g.isPrimaryOutput(v)) period = std::max(period, r.arrival[v]);
    r.period = period;

    // Backward: parallel within each level, from the last level down.
    r.required.assign(g.numNodes, period);
    for (int L = g.numLevels - 1; L >= 0; --L) {
        const int lo = g.levelStart[L], hi = g.levelStart[L + 1];
#pragma omp parallel for schedule(static)
        for (int idx = lo; idx < hi; ++idx) {
            const int v = g.levelNodes[idx];
            const int s = g.foutStart[v], e = g.foutStart[v + 1];
            if (s == e) { r.required[v] = period; continue; }
            float req = std::numeric_limits<float>::infinity();
            for (int p = s; p < e; ++p)
                req = std::min(req, r.required[g.foutTo[p]] - g.foutDelay[p]);
            r.required[v] = req;
        }
    }

    r.slack.resize(g.numNodes);
#pragma omp parallel for schedule(static)
    for (int v = 0; v < g.numNodes; ++v) r.slack[v] = r.required[v] - r.arrival[v];
    return r;
}

}  // namespace egg
