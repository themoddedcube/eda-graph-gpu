// test_oracle_consistency.cpp — the equivalence oracle and staGpu's fallback.
//
//  * maxAbsDiff is a proper zero on identical inputs (staCpu is deterministic).
//  * staGpu, in CPU-fallback mode (no device -> ranGpu=false, as on any g++/CI
//    box), returns *exactly* the CPU reference. On a real GPU host the same call
//    sets ranGpu=true and must still match within float tolerance — we assert
//    the honest contract for whichever path actually ran.
#include <cstdint>

#include "egg_test.h"
#include "sta.h"

using egg::TimingGraph;
using egg::TimingResult;

int main() {
    struct Case { int levels, width, maxFanin; uint64_t seed; };
    const Case cases[] = {
        {6, 5, 3, 1},
        {10, 8, 4, 42},
        {16, 12, 4, 0x9E3779B97F4A7C15ULL},
    };

    for (const Case& c : cases) {
        TimingGraph g = egg::generateLayeredDag(c.levels, c.width, c.maxFanin, c.seed);

        // Oracle self-consistency: same input -> bit-identical outputs -> 0 diff.
        const TimingResult a = egg::staCpu(g);
        const TimingResult b = egg::staCpu(g);
        const double self = egg::maxAbsDiff(a, b);
        CHECK(self == 0.0);

        // staGpu must agree with the CPU oracle.
        bool ranGpu = true;  // sentinel; staGpu overwrites it
        const TimingResult gpu = egg::staGpu(g, &ranGpu);
        const double diff = egg::maxAbsDiff(a, gpu);
        if (ranGpu) {
            // Real device (float kernel): match within tolerance, as main.cpp uses.
            CHECK(diff <= 1e-2);
            std::printf("oracle_consistency: levels=%d width=%d GPU ran, diff=%.3e\n",
                        c.levels, c.width, diff);
        } else {
            // CPU fallback (this box / GH CI): must be an exact reproduction.
            CHECK(diff == 0.0);
            std::printf("oracle_consistency: levels=%d width=%d CPU fallback, diff=%.3e\n",
                        c.levels, c.width, diff);
        }
    }
    std::printf("oracle_consistency: OK\n");
    return 0;
}
