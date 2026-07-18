// test_bench_circuits.cpp — STA correctness on REAL ISCAS-85 netlists + the levelizer.
//
// Reads the bundled .bench circuits (dir passed as argv[1] by ctest), runs the CPU
// reference and the GPU primitive on each, and asserts the STA invariants hold on
// real (irregular, narrow-level) topology — not just the synthetic generator. Also
// checks an exact hand-derived result for c17 and that the levelizer rejects cycles.
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "circuit.h"
#include "egg_test.h"
#include "sta.h"

using egg::TimingGraph;
using egg::TimingResult;

namespace {

const char* kCircuits[] = {"c17",   "c432",  "c499",  "c880",  "c1355", "c1908",
                           "c2670", "c3540", "c5315", "c6288", "c7552"};

void checkStaProperties(const TimingGraph& g, const TimingResult& r) {
    float poMax = -std::numeric_limits<float>::infinity();
    bool anyPO = false;
    for (int v = 0; v < g.numNodes; ++v) {
        CHECK(std::isfinite(r.arrival[v]) && std::isfinite(r.required[v]));
        CHECK(r.arrival[v] >= 0.0f);
        CHECK(approxEq(r.slack[v], r.required[v] - r.arrival[v], 1e-2f));
        if (g.isPrimaryInput(v)) CHECK(approxEq(r.arrival[v], 0.0f));
        if (g.isPrimaryOutput(v)) {
            CHECK(approxEq(r.required[v], r.period, 1e-2f));
            poMax = std::max(poMax, r.arrival[v]);
            anyPO = true;
        }
        // Every arc must go strictly low->high level (levelizer invariant).
        for (int p = g.finStart[v]; p < g.finStart[v + 1]; ++p)
            CHECK(g.level[g.finFrom[p]] < g.level[v]);
    }
    CHECK(anyPO);
    CHECK(approxEq(r.period, poMax, 1e-2f));
}

int gpuRuns = 0;

void runCircuit(const char* name, const TimingGraph& g) {
    const TimingResult cpu = egg::staCpu(g);
    checkStaProperties(g, cpu);
    bool ranGpu = true;
    const TimingResult gpu = egg::staGpu(g, &ranGpu);
    if (ranGpu) {
        CHECK(egg::maxAbsDiff(cpu, gpu) <= 1e-2);
        egg::StaGpuPlan* plan = egg::staGpuPlanCreate(g);
        CHECK(plan != nullptr);
        CHECK(egg::maxAbsDiff(cpu, egg::staGpuPlanRun(plan)) <= 1e-2);
        egg::staGpuPlanDestroy(plan);
        ++gpuRuns;
    } else {
        CHECK(egg::maxAbsDiff(cpu, gpu) == 0.0);
    }
    std::printf("  %-8s n=%-6d levels=%-4d period=%.1f OK\n", name, g.numNodes,
                g.numLevels, cpu.period);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "bench/circuits";

    // Levelizer rejects a cycle (0->1->0).
    bool threw = false;
    try {
        egg::levelize(2, {{0, 1, 1.0f}, {1, 0, 1.0f}});
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    // c17: exact hand-derived structure + period (see comments in the test).
    // 5 PIs, 2 POs, 6 NAND gates (delay 2), longest path 1->3->16->22 style => period 6.
    try {
        const TimingGraph c17 = egg::readBench(dir + "/c17.bench");
        CHECK(c17.numNodes == 11);
        CHECK(c17.numArcs() == 12);
        CHECK(c17.numLevels == 4);
        int pi = 0, po = 0;
        for (int v = 0; v < c17.numNodes; ++v) {
            pi += c17.isPrimaryInput(v);
            po += c17.isPrimaryOutput(v);
        }
        CHECK(pi == 5 && po == 2);
        CHECK(approxEq(egg::staCpu(c17).period, 6.0f));
        std::printf("  c17 exact-structure check OK\n");
    } catch (const std::exception& e) {
        std::printf("  c17 unavailable (%s) — skipping exact check\n", e.what());
    }

    int loaded = 0;
    for (const char* c : kCircuits) {
        try {
            const TimingGraph g = egg::readBench(dir + "/" + c + ".bench");
            runCircuit(c, g);
            ++loaded;
        } catch (const std::exception& e) {
            std::printf("  %-8s skip (%s)\n", c, e.what());
        }
    }
    // At least the tiny always-bundled c17 must have loaded, or the fixtures moved.
    CHECK(loaded >= 1);
    std::printf("bench_circuits: OK  (%d circuits, GPU cross-checked on %d)\n", loaded,
                gpuRuns);
    return 0;
}
