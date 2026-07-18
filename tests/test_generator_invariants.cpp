// test_generator_invariants.cpp — structural invariants of generateLayeredDag.
//
// The generator is the source of every benchmark graph and the oracle input, so
// its output must be a well-formed DAG in CSR: valid fanin prefix, a fanout CSR
// that is the *exact* reversal of the fanin CSR, a level partition covering all
// nodes, and every arc strictly increasing in level (acyclic).
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "egg_test.h"
#include "sta.h"

using egg::TimingGraph;

// Collect every arc as (src, sink, delay) by walking a CSR whose "start" array
// is indexed by the *owner* node and whose neighbour array holds the other end.
// forward=true: owner=sink, neighbour=src (fanin). false: owner=src, neighbour=sink (fanout).
static std::vector<std::tuple<int, int, float>> collectArcs(
    const std::vector<int>& start, const std::vector<int>& neigh,
    const std::vector<float>& delay, int numNodes, bool forwardFanin) {
    std::vector<std::tuple<int, int, float>> arcs;
    for (int owner = 0; owner < numNodes; ++owner) {
        for (int p = start[owner]; p < start[owner + 1]; ++p) {
            const int other = neigh[p];
            const int src = forwardFanin ? other : owner;
            const int snk = forwardFanin ? owner : other;
            arcs.emplace_back(src, snk, delay[p]);
        }
    }
    return arcs;
}

static void checkGraph(const TimingGraph& g) {
    const int n = g.numNodes;
    const int L = g.numLevels;

    // --- fanin CSR prefix is well-formed -----------------------------------
    CHECK(static_cast<int>(g.finStart.size()) == n + 1);
    CHECK(g.finStart[0] == 0);
    for (int v = 0; v < n; ++v) CHECK(g.finStart[v] <= g.finStart[v + 1]);  // non-decreasing
    CHECK(g.finStart[n] == g.numArcs());                                     // ends at numArcs
    CHECK(static_cast<int>(g.foutStart.size()) == n + 1);
    CHECK(g.foutStart[0] == 0);
    for (int v = 0; v < n; ++v) CHECK(g.foutStart[v] <= g.foutStart[v + 1]);
    CHECK(g.foutStart[n] == g.numArcs());

    // --- fanout CSR is the exact reversal of the fanin CSR ------------------
    // Same arc multiset (src, sink, delay) whether read forward or backward.
    auto fin = collectArcs(g.finStart, g.finFrom, g.finDelay, n, /*forwardFanin=*/true);
    auto fout = collectArcs(g.foutStart, g.foutTo, g.foutDelay, n, /*forwardFanin=*/false);
    CHECK(fin.size() == fout.size());
    CHECK(static_cast<int>(fin.size()) == g.numArcs());
    std::sort(fin.begin(), fin.end());
    std::sort(fout.begin(), fout.end());
    CHECK(fin == fout);

    // --- level partition covers every node exactly once --------------------
    CHECK(static_cast<int>(g.levelStart.size()) == L + 1);
    CHECK(static_cast<int>(g.levelNodes.size()) == n);
    CHECK(g.levelStart[0] == 0);
    CHECK(g.levelStart[L] == n);
    for (int l = 0; l < L; ++l) CHECK(g.levelStart[l] <= g.levelStart[l + 1]);
    std::vector<int> seen(n, 0);
    for (int l = 0; l < L; ++l) {
        for (int idx = g.levelStart[l]; idx < g.levelStart[l + 1]; ++idx) {
            const int v = g.levelNodes[idx];
            CHECK(v >= 0 && v < n);
            CHECK(seen[v] == 0);     // each node appears once
            seen[v] = 1;
            CHECK(g.level[v] == l);  // levelNodes bucket matches node's own level
        }
    }
    for (int v = 0; v < n; ++v) CHECK(seen[v] == 1);  // all nodes covered

    // --- acyclic: every arc goes strictly from a lower to a higher level ----
    for (const auto& a : fin) {
        const int src = std::get<0>(a), snk = std::get<1>(a);
        CHECK(g.level[src] < g.level[snk]);
    }
}

int main() {
    struct Case { int levels, width, maxFanin; uint64_t seed; };
    const Case cases[] = {
        {5, 3, 2, 1},
        {8, 6, 3, 42},
        {12, 10, 4, 0x9E3779B97F4A7C15ULL},
        {20, 4, 5, 7},
        {3, 1, 1, 123},  // degenerate: single-node levels
    };
    for (const Case& c : cases) {
        TimingGraph g = egg::generateLayeredDag(c.levels, c.width, c.maxFanin, c.seed);
        CHECK(g.numNodes == c.levels * c.width);
        CHECK(g.numLevels == c.levels);
        checkGraph(g);
        std::printf("generator_invariants: levels=%d width=%d ok (nodes=%d arcs=%d)\n",
                    c.levels, c.width, g.numNodes, g.numArcs());
    }
    std::printf("generator_invariants: OK\n");
    return 0;
}
