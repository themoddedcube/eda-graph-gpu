// test_edge_cases.cpp — base cases, degenerate shapes, and structural properties
// that the primitive must satisfy on ANY valid timing graph.
//
// Strategy:
//   * Hand-built degenerate graphs with arithmetic worked out in the comments:
//     a single node, a single all-PI/PO level, a pure chain, zero-delay arcs, and
//     a high-fanin star (exercises the fanin reduction at degree > 1).
//   * Structural property checks on generated graphs across seeds/sizes — the
//     invariants any correct STA result obeys (PI arrival 0, PO required = period,
//     slack = required - arrival, arrival = max over fanin, period = max PO arrival).
//   * Generator determinism.
//   * Every graph is also cross-checked GPU-vs-CPU when a device is present — both
//     the one-shot staGpu and the persistent CUDA-graph plan — otherwise those
//     branches are skipped honestly (still exit 0 on a CPU-only box / GH CI).
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

#include "egg_test.h"
#include "sta.h"

using egg::TimingGraph;
using egg::TimingResult;

namespace {

struct Arc { int u, v; float d; };

// Build a valid TimingGraph from an explicit level-per-node + arc list. Derives
// the level partition and both CSRs; the fanout CSR is the exact reversal of the
// fanin CSR by construction, matching the generator's invariant.
TimingGraph buildGraph(int n, const std::vector<int>& level,
                       const std::vector<Arc>& arcs) {
    TimingGraph g;
    g.numNodes = n;
    int maxL = 0;
    for (int lv : level) maxL = std::max(maxL, lv);
    g.numLevels = maxL + 1;
    g.level = level;

    g.levelStart.assign(g.numLevels + 1, 0);
    for (int v = 0; v < n; ++v) g.levelStart[level[v] + 1]++;
    for (int l = 0; l < g.numLevels; ++l) g.levelStart[l + 1] += g.levelStart[l];
    g.levelNodes.resize(n);
    {
        std::vector<int> cur(g.levelStart.begin(), g.levelStart.end() - 1);
        for (int v = 0; v < n; ++v) g.levelNodes[cur[level[v]]++] = v;
    }

    g.finStart.assign(n + 1, 0);
    for (const Arc& a : arcs) g.finStart[a.v + 1]++;
    for (int v = 0; v < n; ++v) g.finStart[v + 1] += g.finStart[v];
    g.finFrom.resize(arcs.size());
    g.finDelay.resize(arcs.size());
    {
        std::vector<int> cur(g.finStart.begin(), g.finStart.end() - 1);
        for (const Arc& a : arcs) {
            const int s = cur[a.v]++;
            g.finFrom[s] = a.u;
            g.finDelay[s] = a.d;
        }
    }

    g.foutStart.assign(n + 1, 0);
    for (const Arc& a : arcs) g.foutStart[a.u + 1]++;
    for (int v = 0; v < n; ++v) g.foutStart[v + 1] += g.foutStart[v];
    g.foutTo.resize(arcs.size());
    g.foutDelay.resize(arcs.size());
    {
        std::vector<int> cur(g.foutStart.begin(), g.foutStart.end() - 1);
        for (const Arc& a : arcs) {
            const int s = cur[a.u]++;
            g.foutTo[s] = a.v;
            g.foutDelay[s] = a.d;
        }
    }
    return g;
}

// Cross-check the GPU paths against the CPU reference whenever a device ran.
// Returns true if the GPU actually executed (so callers can report coverage).
bool crossCheckGpu(const TimingGraph& g, const TimingResult& cpu) {
    bool ranGpu = true;  // sentinel; staGpu overwrites
    const TimingResult gpu = egg::staGpu(g, &ranGpu);
    if (!ranGpu) {
        // CPU fallback must be an exact reproduction of the reference.
        CHECK(egg::maxAbsDiff(cpu, gpu) == 0.0);
        return false;
    }
    CHECK(egg::maxAbsDiff(cpu, gpu) <= 1e-2);
    // Persistent CUDA-graph plan must agree too (capture-once / replay).
    egg::StaGpuPlan* plan = egg::staGpuPlanCreate(g);
    CHECK(plan != nullptr);  // a device ran the one-shot, so create must succeed
    const TimingResult a = egg::staGpuPlanRun(plan);
    const TimingResult b = egg::staGpuPlanRun(plan);  // replay is idempotent
    egg::staGpuPlanDestroy(plan);
    CHECK(egg::maxAbsDiff(cpu, a) <= 1e-2);
    CHECK(egg::maxAbsDiff(a, b) == 0.0);
    return true;
}

// The invariants any correct STA result must satisfy on any graph.
void checkProperties(const TimingGraph& g, const TimingResult& r) {
    const int n = g.numNodes;
    CHECK(static_cast<int>(r.arrival.size()) == n);
    CHECK(static_cast<int>(r.required.size()) == n);
    CHECK(static_cast<int>(r.slack.size()) == n);

    float poMax = -std::numeric_limits<float>::infinity();
    bool anyPO = false;
    for (int v = 0; v < n; ++v) {
        CHECK(std::isfinite(r.arrival[v]));
        CHECK(std::isfinite(r.required[v]));
        CHECK(r.arrival[v] >= 0.0f);
        // slack is exactly required - arrival.
        CHECK(approxEq(r.slack[v], r.required[v] - r.arrival[v], 1e-3f));
        // PI arrives at 0.
        if (g.isPrimaryInput(v)) CHECK(approxEq(r.arrival[v], 0.0f));
        // PO's required time is the period.
        if (g.isPrimaryOutput(v)) {
            CHECK(approxEq(r.required[v], r.period, 1e-3f));
            poMax = std::max(poMax, r.arrival[v]);
            anyPO = true;
        }
        // arrival = max over fanin of (arrival[u] + delay); recomputed independently.
        const int s = g.finStart[v], e = g.finStart[v + 1];
        if (s != e) {
            float a = -std::numeric_limits<float>::infinity();
            for (int p = s; p < e; ++p)
                a = std::max(a, r.arrival[g.finFrom[p]] + g.finDelay[p]);
            CHECK(approxEq(r.arrival[v], a, 1e-3f));
        }
    }
    CHECK(anyPO);
    // period is exactly the max arrival over primary outputs.
    CHECK(approxEq(r.period, poMax, 1e-3f));
}

int gpuRuns = 0;

void runCase(const char* name, const TimingGraph& g,
             const std::vector<float>& expArrival,
             const std::vector<float>& expRequired,
             const std::vector<float>& expSlack, float expPeriod) {
    const TimingResult r = egg::staCpu(g);
    CHECK(approxEq(r.period, expPeriod));
    for (int v = 0; v < g.numNodes; ++v) {
        CHECK(approxEq(r.arrival[v], expArrival[v]));
        CHECK(approxEq(r.required[v], expRequired[v]));
        CHECK(approxEq(r.slack[v], expSlack[v]));
    }
    checkProperties(g, r);
    if (crossCheckGpu(g, r)) ++gpuRuns;
    std::printf("  edge: %-22s OK (period=%.1f)\n", name, r.period);
}

}  // namespace

int main() {
    // 1) Single node: it is simultaneously a PI and a PO. arrival=required=slack=0.
    runCase("single_node",
            buildGraph(1, {0}, {}),
            {0}, {0}, {0}, /*period*/ 0.0f);

    // 2) One level of three isolated nodes: all PI+PO, everything zero.
    runCase("single_level_trio",
            buildGraph(3, {0, 0, 0}, {}),
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 0.0f);

    // 3) Pure chain 0->1->2->3 with delays 1,2,3.
    //    arrival 0,1,3,6 ; period 6 ; single path -> all slack 0.
    runCase("chain_of_4",
            buildGraph(4, {0, 1, 2, 3}, {{0, 1, 1}, {1, 2, 2}, {2, 3, 3}}),
            {0, 1, 3, 6}, {0, 1, 3, 6}, {0, 0, 0, 0}, 6.0f);

    // 4) Zero-delay arc: 0->1(0), 0->2(5). POs {1,2}. period=max(0,5)=5.
    //    required r1=5,r2=5,r0=min(5-0,5-5)=0. slack 0,5,0.
    runCase("zero_delay_arc",
            buildGraph(3, {0, 1, 1}, {{0, 1, 0}, {0, 2, 5}}),
            {0, 0, 5}, {0, 5, 5}, {0, 5, 0}, 5.0f);

    // 5) High-fanin star: PIs 0..3 -> PO 4 with delays 1,2,3,4.
    //    arrival4=max(1,2,3,4)=4 ; period 4 ; required PIs = 4-delay.
    runCase("high_fanin_star",
            buildGraph(5, {0, 0, 0, 0, 1},
                       {{0, 4, 1}, {1, 4, 2}, {2, 4, 3}, {3, 4, 4}}),
            {0, 0, 0, 0, 4}, {3, 2, 1, 0, 4}, {3, 2, 1, 0, 0}, 4.0f);

    // 6) Structural properties + GPU cross-check on generated graphs, including a
    //    degenerate 1-wide (chain-like) and a wide/high-fanin shape.
    struct Case { int levels, width, maxFanin; uint64_t seed; };
    const Case cases[] = {
        {1, 1, 1, 7},           // single node via the generator
        {2, 1, 1, 7},           // 1-wide (pure chain)
        {5, 3, 2, 11},
        {12, 8, 4, 42},
        {20, 50, 8, 0xDEADBEEF},  // high fanin -> load imbalance
        {64, 256, 4, 123},
    };
    for (const Case& c : cases) {
        const TimingGraph g =
            egg::generateLayeredDag(c.levels, c.width, c.maxFanin, c.seed);
        const TimingResult r = egg::staCpu(g);
        checkProperties(g, r);
        if (crossCheckGpu(g, r)) ++gpuRuns;
        std::printf("  gen : L=%-3d W=%-4d fin<=%d  n=%-7d arcs=%-8d period=%.2f OK\n",
                    c.levels, c.width, c.maxFanin, g.numNodes, g.numArcs(), r.period);
    }

    // 7) Generator determinism: same seed -> identical CSR + identical result.
    {
        const TimingGraph a = egg::generateLayeredDag(16, 32, 4, 999);
        const TimingGraph b = egg::generateLayeredDag(16, 32, 4, 999);
        CHECK(a.numNodes == b.numNodes && a.numArcs() == b.numArcs());
        CHECK(a.finFrom == b.finFrom && a.finDelay == b.finDelay);
        CHECK(a.foutTo == b.foutTo && a.foutDelay == b.foutDelay);
        CHECK(egg::maxAbsDiff(egg::staCpu(a), egg::staCpu(b)) == 0.0);
        std::printf("  determinism: identical graph + result for a fixed seed OK\n");
    }

    std::printf("edge_cases: OK  (GPU cross-checked on %d graphs%s)\n", gpuRuns,
                gpuRuns ? "" : " — none; CPU-only box, GPU paths skipped honestly");
    return 0;
}
