// circuit.h — build timing graphs from REAL netlists, and profile their topology.
//
// The synthetic generator (generateLayeredDag) produces pre-levelized, uniform-width
// graphs. Real circuits don't: they arrive as a gate netlist with no levels, with
// irregular level widths and skewed fanin/fanout. This header adds (1) a general DAG
// levelizer that turns an arbitrary arc list into a valid level-parallel TimingGraph
// (and rejects cycles), (2) an ISCAS-85 `.bench` reader on top of it, and (3) a
// topology profiler so real vs synthetic level/degree distributions can be compared.
#pragma once

#include <string>
#include <vector>

#include "timing_graph.h"

namespace egg {

struct Arc {
    int from;
    int to;
    float delay;
};

// Levelize an arbitrary DAG (numNodes nodes, `arcs` from->to with delays) into a
// TimingGraph: level = longest path from a source, so every arc goes strictly
// low->high level (the invariant the STA sweeps need). Throws std::runtime_error on
// a cycle. `endpoints` are node ids that must be primary outputs; a zero-delay
// virtual sink is appended for any endpoint that still has fanout, so the
// "PO == node with no fanout" model matches the circuit's real outputs.
TimingGraph levelize(int numNodes, std::vector<Arc> arcs,
                     const std::vector<int>& endpoints = {});

// Read an ISCAS-85 combinational `.bench` netlist into a levelized TimingGraph.
// Gate arcs carry a nominal unit-ish delay model (BUFF/NOT=1, XOR=3, else 2) — real
// STA uses a cell library; this is a documented stand-in, labeled as such.
TimingGraph readBench(const std::string& path);

// Topology summary for comparing real circuits against the synthetic generator.
struct GraphProfile {
    int numNodes = 0, numLevels = 0, numArcs = 0;
    int minLevelWidth = 0, maxLevelWidth = 0;
    double meanLevelWidth = 0.0;
    int maxFanin = 0, maxFanout = 0;
    double meanFanin = 0.0;
    int numPI = 0, numPO = 0;
    float period = 0.0f;  // critical-path length (from staCpu)
};

GraphProfile profileGraph(const TimingGraph& g);

}  // namespace egg
