// circuit.cpp — DAG levelizer, ISCAS-85 .bench reader, topology profiler.
#include "circuit.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include "sta.h"

namespace egg {

TimingGraph levelize(int numNodes, std::vector<Arc> arcs,
                     const std::vector<int>& endpoints) {
    int n = numNodes;
    // Append a zero-delay virtual sink for any endpoint that still drives a gate, so
    // "primary output == node with no fanout" matches the circuit's declared outputs.
    if (!endpoints.empty()) {
        std::vector<char> hasFanout(n, 0);
        for (const Arc& a : arcs) hasFanout[a.from] = 1;
        for (int e : endpoints) {
            if (e >= 0 && e < static_cast<int>(hasFanout.size()) && hasFanout[e])
                arcs.push_back({e, n++, 0.0f});
        }
    }

    // Longest-path levelization via Kahn topological order (detects cycles).
    std::vector<int> indeg(n, 0), level(n, 0);
    std::vector<std::vector<int>> fout(n);
    for (const Arc& a : arcs) {
        indeg[a.to]++;
        fout[a.from].push_back(a.to);
    }
    std::vector<int> q;
    for (int v = 0; v < n; ++v)
        if (indeg[v] == 0) q.push_back(v);
    std::vector<int> rem = indeg;
    int processed = 0;
    for (size_t h = 0; h < q.size(); ++h) {
        const int u = q[h];
        ++processed;
        for (int w : fout[u]) {
            level[w] = std::max(level[w], level[u] + 1);
            if (--rem[w] == 0) q.push_back(w);
        }
    }
    if (processed != n)
        throw std::runtime_error("levelize: graph is not a DAG (cycle detected)");

    int numLevels = 0;
    for (int v = 0; v < n; ++v) numLevels = std::max(numLevels, level[v] + 1);

    TimingGraph g;
    g.numNodes = n;
    g.numLevels = numLevels;
    g.level = level;

    g.levelStart.assign(numLevels + 1, 0);
    for (int v = 0; v < n; ++v) g.levelStart[level[v] + 1]++;
    for (int l = 0; l < numLevels; ++l) g.levelStart[l + 1] += g.levelStart[l];
    g.levelNodes.resize(n);
    {
        std::vector<int> cur(g.levelStart.begin(), g.levelStart.end() - 1);
        for (int v = 0; v < n; ++v) g.levelNodes[cur[level[v]]++] = v;
    }

    // Fanin CSR, arcs grouped by sink node.
    g.finStart.assign(n + 1, 0);
    for (const Arc& a : arcs) g.finStart[a.to + 1]++;
    for (int v = 0; v < n; ++v) g.finStart[v + 1] += g.finStart[v];
    g.finFrom.resize(arcs.size());
    g.finDelay.resize(arcs.size());
    {
        std::vector<int> cur(g.finStart.begin(), g.finStart.end() - 1);
        for (const Arc& a : arcs) {
            const int s = cur[a.to]++;
            g.finFrom[s] = a.from;
            g.finDelay[s] = a.delay;
        }
    }

    // Fanout CSR = reversal of the fanin arcs.
    g.foutStart.assign(n + 1, 0);
    for (int p = 0; p < g.numArcs(); ++p) g.foutStart[g.finFrom[p] + 1]++;
    for (int v = 0; v < n; ++v) g.foutStart[v + 1] += g.foutStart[v];
    g.foutTo.resize(arcs.size());
    g.foutDelay.resize(arcs.size());
    {
        std::vector<int> cur(g.foutStart.begin(), g.foutStart.end() - 1);
        for (int v = 0; v < n; ++v)
            for (int p = g.finStart[v]; p < g.finStart[v + 1]; ++p) {
                const int u = g.finFrom[p];
                const int s = cur[u]++;
                g.foutTo[s] = v;
                g.foutDelay[s] = g.finDelay[p];
            }
    }
    return g;
}

namespace {

float gateDelay(const std::string& type) {
    if (type == "BUFF" || type == "BUF" || type == "NOT") return 1.0f;
    if (type == "XOR" || type == "XNOR") return 3.0f;
    return 2.0f;  // AND / OR / NAND / NOR
}

std::string stripSpaces(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') out += c;
    return out;
}

}  // namespace

TimingGraph readBench(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("readBench: cannot open " + path);

    std::unordered_map<std::string, int> id;
    auto getId = [&](const std::string& s) -> int {
        auto it = id.find(s);
        if (it != id.end()) return it->second;
        const int v = static_cast<int>(id.size());
        id.emplace(s, v);
        return v;
    };

    std::vector<int> outputs;
    std::vector<Arc> arcs;
    std::string raw;
    while (std::getline(in, raw)) {
        const auto hash = raw.find('#');
        std::string line = stripSpaces(hash == std::string::npos ? raw : raw.substr(0, hash));
        if (line.empty()) continue;

        if (line.rfind("INPUT(", 0) == 0) {
            getId(line.substr(6, line.find(')') - 6));  // ensure the PI node exists
        } else if (line.rfind("OUTPUT(", 0) == 0) {
            outputs.push_back(getId(line.substr(7, line.find(')') - 7)));
        } else {
            const auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            const int out = getId(line.substr(0, eq));
            const std::string rhs = line.substr(eq + 1);
            const auto lp = rhs.find('('), rp = rhs.find(')');
            if (lp == std::string::npos || rp == std::string::npos) continue;
            const float d = gateDelay(rhs.substr(0, lp));
            const std::string args = rhs.substr(lp + 1, rp - lp - 1);
            size_t start = 0;
            while (start <= args.size()) {
                const size_t comma = args.find(',', start);
                const std::string a =
                    args.substr(start, comma == std::string::npos ? std::string::npos
                                                                  : comma - start);
                if (!a.empty()) arcs.push_back({getId(a), out, d});
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
    }
    return levelize(static_cast<int>(id.size()), std::move(arcs), outputs);
}

GraphProfile profileGraph(const TimingGraph& g) {
    GraphProfile p;
    p.numNodes = g.numNodes;
    p.numLevels = g.numLevels;
    p.numArcs = g.numArcs();

    p.minLevelWidth = g.numNodes;
    for (int l = 0; l < g.numLevels; ++l) {
        const int w = g.levelStart[l + 1] - g.levelStart[l];
        p.maxLevelWidth = std::max(p.maxLevelWidth, w);
        p.minLevelWidth = std::min(p.minLevelWidth, w);
    }
    p.meanLevelWidth = g.numLevels ? static_cast<double>(g.numNodes) / g.numLevels : 0.0;

    for (int v = 0; v < g.numNodes; ++v) {
        p.maxFanin = std::max(p.maxFanin, g.finStart[v + 1] - g.finStart[v]);
        p.maxFanout = std::max(p.maxFanout, g.foutStart[v + 1] - g.foutStart[v]);
        if (g.isPrimaryInput(v)) ++p.numPI;
        if (g.isPrimaryOutput(v)) ++p.numPO;
    }
    p.meanFanin = g.numNodes ? static_cast<double>(g.numArcs()) / g.numNodes : 0.0;
    p.period = staCpu(g).period;
    return p;
}

TimingGraph withFinDelay(const TimingGraph& g, const std::vector<float>& newFinDelay) {
    TimingGraph r = g;  // copy topology, levels, and CSR index arrays
    r.finDelay = newFinDelay;
    // Rebuild foutDelay by the SAME reversal mapping that produced g.foutTo, so each
    // fanout slot gets the updated delay of its arc.
    std::vector<int> cur(g.foutStart.begin(), g.foutStart.end() - 1);
    for (int v = 0; v < g.numNodes; ++v)
        for (int p = g.finStart[v]; p < g.finStart[v + 1]; ++p)
            r.foutDelay[cur[g.finFrom[p]]++] = newFinDelay[p];
    return r;
}

}  // namespace egg
