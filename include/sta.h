// sta.h — static timing analysis primitive (CPU reference + GPU).
#pragma once

#include <cstdint>
#include <vector>

#include "timing_graph.h"

namespace egg {

struct TimingResult {
    std::vector<float> arrival;   // latest arrival time at each node
    std::vector<float> required;  // required time at each node
    std::vector<float> slack;     // required - arrival
    float period = 0.0f;          // critical-path length (max PO arrival)
};

// CPU reference: level-ordered forward (arrival) + backward (required). O(V+E).
// This is the oracle the GPU primitive is checked against.
TimingResult staCpu(const TimingGraph& g);

// Multi-threaded CPU reference (OpenMP): the SAME level decomposition as the GPU
// (nodes within a level in parallel, levels serial). A FAIR all-core CPU baseline
// for the GPU comparison — not just a single thread. numThreads <= 0 uses the
// OpenMP default (all cores). Bit-identical to staCpu (asserted in the tests).
TimingResult staCpuParallel(const TimingGraph& g, int numThreads = 0);

// GPU level-parallel STA primitive: one kernel per topological level, all nodes
// in a level updated in parallel. Falls back to the CPU reference and sets
// *ranGpu=false when no CUDA device is present, so the same binary runs (and
// stays honest) with or without a GPU.
TimingResult staGpu(const TimingGraph& g, bool* ranGpu);

// --- Persistent GPU plan (improvement iteration 2: CUDA-graph capture) ---
// Real STA re-evaluates the same topology many times (incremental timing inside a
// placement / optimization loop). staGpuPlanCreate uploads the graph and captures
// the entire level-sweep (forward + on-device period reduce + backward) into a CUDA
// graph ONCE; staGpuPlanRun replays it with a single cudaGraphLaunch, amortizing the
// ~2*numLevels per-level kernel-launch latencies that dominate the naive path.
// Create returns nullptr when there is no CUDA device (callers fall back to staCpu).
// The captured math is identical to staCpu, so the maxAbsDiff oracle still holds.
struct StaGpuPlan;
StaGpuPlan* staGpuPlanCreate(const TimingGraph& g);
TimingResult staGpuPlanRun(StaGpuPlan* plan);
void staGpuPlanDestroy(StaGpuPlan* plan);

// Order-independent equivalence check: the max abs difference of the arrival
// and required arrays (STA outputs are deterministic value arrays, so this is
// the right comparator — not a path/order diff).
double maxAbsDiff(const TimingResult& a, const TimingResult& b);

// Deterministic synthetic layered-DAG timing graph: `numLevels` levels of
// `widthPerLevel` nodes; each non-source node draws 1..maxFaninPerNode fanin
// arcs from the preceding one or two levels, with random delays.
TimingGraph generateLayeredDag(int numLevels, int widthPerLevel,
                               int maxFaninPerNode, uint64_t seed);

}  // namespace egg
