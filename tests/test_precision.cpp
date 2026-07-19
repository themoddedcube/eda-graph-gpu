// test_precision.cpp — bound the fp32 STA error against a double-precision truth.
//
// maxAbsDiff (fp32 vs fp32) is self-referential. This checks the fp32 CPU reference
// AND the GPU against staCpuDouble (the ground truth for these exact float inputs),
// over DEEP graphs (hundreds of levels of accumulation) where fp32 error is worst,
// and asserts the error stays within a small, explicit bound — including slack near
// zero, the timing-critical, cancellation-prone region.
#include <cstdint>

#include "egg_test.h"
#include "sta.h"

using egg::TimingGraph;
using egg::TimingResult;
using egg::TimingResultD;

int main() {
    // Deep + wide graphs: many levels => long accumulation chains for arrival.
    struct Case { int levels, width; uint64_t seed; };
    const Case cases[] = {{400, 2000, 7}, {800, 4000, 99}, {1200, 1000, 3}};

    for (const Case& c : cases) {
        const TimingGraph g = egg::generateLayeredDag(c.levels, c.width, 4, c.seed);
        const TimingResult f = egg::staCpu(g);
        const TimingResultD d = egg::staCpuDouble(g);
        const egg::Fp32Error e = egg::fp32Error(f, d);

        // Explicit accuracy bounds (calibrated to measured fp32 behavior, with margin):
        //  - relative arrival error is tiny (fp32 has ~7 digits; period ~ thousands),
        //  - absolute slack error near zero stays well below any real timing margin.
        CHECK(e.maxRelArrival < 1e-4);
        CHECK(e.worstSlackAbsNearZero < 1e-1);
        CHECK(e.maxAbsSlack < 1.0);

        std::printf("  L=%-4d W=%-4d period=%.1f  fp32 err: relArr=%.2e absSlack=%.2e "
                    "nearZeroSlack=%.2e\n", c.levels, c.width, d.period,
                    e.maxRelArrival, e.maxAbsSlack, e.worstSlackAbsNearZero);

        // If a GPU ran, its fp32 result must be just as accurate vs the truth.
        bool ranGpu = false;
        const TimingResult gpu = egg::staGpu(g, &ranGpu);
        if (ranGpu) {
            const egg::Fp32Error ge = egg::fp32Error(gpu, d);
            CHECK(ge.maxRelArrival < 1e-4);
            CHECK(ge.worstSlackAbsNearZero < 1e-1);
        }
    }
    std::printf("precision: OK (fp32 bounded against fp64 truth)\n");
    return 0;
}
