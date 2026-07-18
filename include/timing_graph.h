// timing_graph.h — SoA timing graph for level-parallel static timing analysis.
//
// A timing graph is a DAG: nodes are pins, edges are timing arcs with delays.
// STA propagates arrival times forward (max over fanin) and required times
// backward (min over fanout). Nodes at the same topological LEVEL are mutually
// independent — that is the parallelism this library's GPU primitive exploits.
//
// Everything is struct-of-arrays / CSR so the CPU reference and the CUDA kernel
// share one layout (host arrays <-> device arrays, no marshalling gap).
#pragma once

#include <cstdint>
#include <vector>

namespace egg {

struct TimingGraph {
    int numNodes = 0;
    int numLevels = 0;

    // Topological level of each node (0 = primary input / source).
    std::vector<int> level;          // [numNodes]

    // Nodes grouped by level: level L occupies
    // levelNodes[levelStart[L] .. levelStart[L+1]).
    std::vector<int> levelStart;     // [numLevels + 1]
    std::vector<int> levelNodes;     // [numNodes]

    // Fanin (incoming) arcs in CSR — used by the forward arrival pass.
    std::vector<int> finStart;       // [numNodes + 1]
    std::vector<int> finFrom;        // [numArcs]  source node of the arc
    std::vector<float> finDelay;     // [numArcs]  arc delay

    // Fanout (outgoing) arcs in CSR — used by the backward required-time pass.
    std::vector<int> foutStart;      // [numNodes + 1]
    std::vector<int> foutTo;         // [numArcs]  sink node of the arc
    std::vector<float> foutDelay;    // [numArcs]

    int numArcs() const { return static_cast<int>(finFrom.size()); }
    bool isPrimaryInput(int v) const { return finStart[v] == finStart[v + 1]; }
    bool isPrimaryOutput(int v) const { return foutStart[v] == foutStart[v + 1]; }
};

}  // namespace egg
