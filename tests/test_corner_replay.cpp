// test_corner_replay.cpp — the CUDA-graph plan re-evaluated under CHANGING delays.
//
// This is the honest answer to "replay just recomputes a bit-identical answer": a
// plan is captured once, then swapped through several timing CORNERS (new arc delays
// over the same topology) via staGpuPlanUpdateDelays. For each corner the GPU result
// must (a) match the CPU reference on THAT corner (correct on changing input) and
// (b) actually differ from the base corner (proof it is not re-running one answer).
#include <cstdint>
#include <random>
#include <vector>

#include "circuit.h"
#include "egg_test.h"
#include "sta.h"

using egg::TimingGraph;
using egg::TimingResult;

// A non-uniform per-arc delay corner: each arc scaled by a random factor in [0.5,1.5].
static std::vector<float> cornerDelays(const TimingGraph& g, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> scale(0.5f, 1.5f);
    std::vector<float> d(g.finDelay.size());
    for (size_t i = 0; i < d.size(); ++i) d[i] = g.finDelay[i] * scale(rng);
    return d;
}

int main() {
    const TimingGraph base = egg::generateLayeredDag(64, 256, 4, 12345);

    // withFinDelay + staCpu must model a real corner on the CPU side regardless of GPU.
    {
        const std::vector<float> d1 = cornerDelays(base, 1);
        const TimingGraph g1 = egg::withFinDelay(base, d1);
        CHECK(g1.finDelay == d1);
        const double changed = egg::maxAbsDiff(egg::staCpu(base), egg::staCpu(g1));
        CHECK(changed > 0.5);  // a different corner really is a different answer
    }

    egg::StaGpuPlan* plan = egg::staGpuPlanCreate(base);
    if (!plan) {
        std::printf("corner_replay: no CUDA device — CPU corner check only, OK\n");
        return 0;
    }

    // Base corner: plan replay == CPU reference on the base delays.
    const TimingResult cpuBase = egg::staCpu(base);
    const TimingResult r0 = egg::staGpuPlanRun(plan);
    CHECK(egg::maxAbsDiff(cpuBase, r0) <= 1e-2);

    // A wrong-size update must be rejected, and must NOT corrupt the plan.
    CHECK(!egg::staGpuPlanUpdateDelays(plan, {1.0f, 2.0f}, {1.0f, 2.0f}));
    CHECK(egg::maxAbsDiff(cpuBase, egg::staGpuPlanRun(plan)) <= 1e-2);

    int corners = 0;
    for (uint64_t c = 1; c <= 6; ++c) {
        const std::vector<float> d = cornerDelays(base, c);
        const TimingGraph gc = egg::withFinDelay(base, d);
        CHECK(egg::staGpuPlanUpdateDelays(plan, gc.finDelay, gc.foutDelay));
        const TimingResult rc = egg::staGpuPlanRun(plan);
        // (a) correct on the changed input:
        CHECK(egg::maxAbsDiff(egg::staCpu(gc), rc) <= 1e-2);
        // (b) genuinely a different result from the base corner:
        CHECK(egg::maxAbsDiff(r0, rc) > 0.5);
        ++corners;
    }

    egg::staGpuPlanDestroy(plan);
    std::printf("corner_replay: OK  (%d corners, each GPU==CPU on its own delays and "
                "!= base)\n", corners);
    return 0;
}
