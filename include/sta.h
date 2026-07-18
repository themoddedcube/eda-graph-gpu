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

// GPU level-parallel STA primitive: one kernel per topological level, all nodes
// in a level updated in parallel. Falls back to the CPU reference and sets
// *ranGpu=false when no CUDA device is present, so the same binary runs (and
// stays honest) with or without a GPU.
TimingResult staGpu(const TimingGraph& g, bool* ranGpu);

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
