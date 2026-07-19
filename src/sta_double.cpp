// sta_double.cpp — double-precision STA ground truth + fp32 error bound.
//
// The maxAbsDiff oracle proves the GPU reproduces the fp32 CPU reference, but both
// round the same way, so it says nothing about ACCURACY. This computes the same STA
// in double precision (the ground truth for these exact float inputs) and bounds the
// fp32 error — including the catastrophic-cancellation-prone slack near zero, over the
// hundreds of levels of accumulation a deep timing graph has.
#include <algorithm>
#include <cmath>
#include <limits>

#include "sta.h"

namespace egg {

TimingResultD staCpuDouble(const TimingGraph& g) {
    TimingResultD r;
    r.arrival.assign(g.numNodes, 0.0);
    for (int L = 0; L < g.numLevels; ++L) {
        for (int idx = g.levelStart[L]; idx < g.levelStart[L + 1]; ++idx) {
            const int v = g.levelNodes[idx];
            const int s = g.finStart[v], e = g.finStart[v + 1];
            if (s == e) { r.arrival[v] = 0.0; continue; }
            double a = -std::numeric_limits<double>::infinity();
            for (int p = s; p < e; ++p)
                a = std::max(a, r.arrival[g.finFrom[p]] +
                                    static_cast<double>(g.finDelay[p]));
            r.arrival[v] = a;
        }
    }
    double period = 0.0;
    for (int v = 0; v < g.numNodes; ++v)
        if (g.isPrimaryOutput(v)) period = std::max(period, r.arrival[v]);
    r.period = period;

    r.required.assign(g.numNodes, period);
    for (int L = g.numLevels - 1; L >= 0; --L) {
        for (int idx = g.levelStart[L]; idx < g.levelStart[L + 1]; ++idx) {
            const int v = g.levelNodes[idx];
            const int s = g.foutStart[v], e = g.foutStart[v + 1];
            if (s == e) { r.required[v] = period; continue; }
            double req = std::numeric_limits<double>::infinity();
            for (int p = s; p < e; ++p)
                req = std::min(req, r.required[g.foutTo[p]] -
                                        static_cast<double>(g.foutDelay[p]));
            r.required[v] = req;
        }
    }
    r.slack.resize(g.numNodes);
    for (int v = 0; v < g.numNodes; ++v) r.slack[v] = r.required[v] - r.arrival[v];
    return r;
}

Fp32Error fp32Error(const TimingResult& f, const TimingResultD& d) {
    Fp32Error e{};
    const int n = static_cast<int>(f.arrival.size());
    for (int i = 0; i < n; ++i) {
        const double aErr = std::abs(static_cast<double>(f.arrival[i]) - d.arrival[i]);
        const double rErr = std::abs(static_cast<double>(f.required[i]) - d.required[i]);
        const double sErr = std::abs(static_cast<double>(f.slack[i]) - d.slack[i]);
        e.maxAbsArrival = std::max(e.maxAbsArrival, aErr);
        e.maxAbsRequired = std::max(e.maxAbsRequired, rErr);
        e.maxAbsSlack = std::max(e.maxAbsSlack, sErr);
        if (d.arrival[i] > 1.0)
            e.maxRelArrival = std::max(e.maxRelArrival, aErr / d.arrival[i]);
        // Slack near zero (the timing-critical region) is cancellation-prone; track
        // the worst absolute error there separately.
        if (std::abs(d.slack[i]) < 1.0)
            e.worstSlackAbsNearZero = std::max(e.worstSlackAbsNearZero, sErr);
    }
    return e;
}

}  // namespace egg
