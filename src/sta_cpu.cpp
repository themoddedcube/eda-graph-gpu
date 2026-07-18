// sta_cpu.cpp — CPU reference STA, the equivalence oracle, and the synthetic
// timing-graph generator.
#include "sta.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace egg {

TimingResult staCpu(const TimingGraph& g) {
    TimingResult r;
    r.arrival.assign(g.numNodes, 0.0f);

    // Forward pass: process levels 0..L-1; a node's arrival is the max over its
    // fanin of (fanin arrival + arc delay). Primary inputs (no fanin) arrive 0.
    for (int L = 0; L < g.numLevels; ++L) {
        for (int idx = g.levelStart[L]; idx < g.levelStart[L + 1]; ++idx) {
            const int v = g.levelNodes[idx];
            const int s = g.finStart[v], e = g.finStart[v + 1];
            if (s == e) { r.arrival[v] = 0.0f; continue; }
            float a = -std::numeric_limits<float>::infinity();
            for (int p = s; p < e; ++p)
                a = std::max(a, r.arrival[g.finFrom[p]] + g.finDelay[p]);
            r.arrival[v] = a;
        }
    }

    // Critical-path length = max arrival over primary outputs.
    float period = 0.0f;
    for (int v = 0; v < g.numNodes; ++v)
        if (g.isPrimaryOutput(v)) period = std::max(period, r.arrival[v]);
    r.period = period;

    // Backward pass: required[PO] = period; otherwise min over fanout of
    // (fanout required - arc delay).
    r.required.assign(g.numNodes, period);
    for (int L = g.numLevels - 1; L >= 0; --L) {
        for (int idx = g.levelStart[L]; idx < g.levelStart[L + 1]; ++idx) {
            const int v = g.levelNodes[idx];
            const int s = g.foutStart[v], e = g.foutStart[v + 1];
            if (s == e) { r.required[v] = period; continue; }
            float req = std::numeric_limits<float>::infinity();
            for (int p = s; p < e; ++p)
                req = std::min(req, r.required[g.foutTo[p]] - g.foutDelay[p]);
            r.required[v] = req;
        }
    }

    r.slack.resize(g.numNodes);
    for (int v = 0; v < g.numNodes; ++v) r.slack[v] = r.required[v] - r.arrival[v];
    return r;
}

double maxAbsDiff(const TimingResult& a, const TimingResult& b) {
    double d = 0.0;
    const int n = static_cast<int>(a.arrival.size());
    for (int i = 0; i < n; ++i) {
        d = std::max(d, std::abs(static_cast<double>(a.arrival[i]) - b.arrival[i]));
        d = std::max(d, std::abs(static_cast<double>(a.required[i]) - b.required[i]));
    }
    return d;
}

TimingGraph generateLayeredDag(int numLevels, int widthPerLevel,
                               int maxFaninPerNode, uint64_t seed) {
    TimingGraph g;
    g.numLevels = numLevels;
    g.numNodes = numLevels * widthPerLevel;
    // Node id = L*width + w, so nodes are already grouped by level in id order.
    g.level.resize(g.numNodes);
    g.levelStart.resize(numLevels + 1);
    g.levelNodes.resize(g.numNodes);
    for (int L = 0; L <= numLevels; ++L) g.levelStart[L] = L * widthPerLevel;
    for (int v = 0; v < g.numNodes; ++v) {
        g.level[v] = v / widthPerLevel;
        g.levelNodes[v] = v;
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> delayDist(0.5f, 5.0f);

    // Build fanin arcs per node, then derive the fanout CSR by reversal.
    g.finStart.assign(g.numNodes + 1, 0);
    std::vector<int> finFrom;
    std::vector<float> finDelay;
    for (int L = 1; L < numLevels; ++L) {
        const int loLevel = std::max(0, L - 2);
        for (int w = 0; w < widthPerLevel; ++w) {
            const int v = L * widthPerLevel + w;
            const int lo = loLevel * widthPerLevel;
            const int hi = L * widthPerLevel;  // exclusive: only lower levels
            const int span = hi - lo;
            const int fanin = 1 + static_cast<int>(rng() % maxFaninPerNode);
            for (int k = 0; k < fanin; ++k) {
                const int u = lo + static_cast<int>(rng() % span);
                finFrom.push_back(u);
                finDelay.push_back(delayDist(rng));
            }
            g.finStart[v + 1] = static_cast<int>(finFrom.size());
        }
    }
    // finStart currently holds per-node end offsets only where set; make it a
    // proper prefix by carrying forward zeros for the (empty-fanin) PIs.
    for (int v = 1; v <= g.numNodes; ++v)
        if (g.finStart[v] < g.finStart[v - 1]) g.finStart[v] = g.finStart[v - 1];
    g.finFrom = std::move(finFrom);
    g.finDelay = std::move(finDelay);

    // Fanout CSR = reversal of the fanin arcs.
    g.foutStart.assign(g.numNodes + 1, 0);
    for (int p = 0; p < g.numArcs(); ++p) g.foutStart[g.finFrom[p] + 1]++;
    for (int v = 0; v < g.numNodes; ++v) g.foutStart[v + 1] += g.foutStart[v];
    g.foutTo.resize(g.numArcs());
    g.foutDelay.resize(g.numArcs());
    std::vector<int> cursor(g.foutStart.begin(), g.foutStart.end() - 1);
    for (int v = 0; v < g.numNodes; ++v) {
        for (int p = g.finStart[v]; p < g.finStart[v + 1]; ++p) {
            const int u = g.finFrom[p];          // arc u -> v
            const int slot = cursor[u]++;
            g.foutTo[slot] = v;
            g.foutDelay[slot] = g.finDelay[p];
        }
    }
    return g;
}

}  // namespace egg
