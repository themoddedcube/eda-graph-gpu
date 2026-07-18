// test_hand_computed.cpp — correctness of staCpu against a DAG solved on paper.
//
// Graph (5 nodes, 3 levels). Arc labels are delays:
//
//        (2)      (4)
//    0 ------> 1 ------> 3        POs: 3, 4   (no fanout)
//     \                 ^
//   (3)\      (1)      /
//       > 2 ---------/
//         \  (5)
//          -------> 4
//
// Fanin arcs:  1<-0(2)   2<-0(3)   3<-1(4)   3<-2(1)   4<-2(5)
//
// Hand computation
//   arrival:   a0=0                       (primary input)
//              a1=a0+2=2   a2=a0+3=3
//              a3=max(a1+4, a2+1)=max(6,4)=6
//              a4=a2+5=8
//   period  =  max over POs {a3,a4} = max(6,8) = 8
//   required:  r3=r4=period=8             (primary outputs)
//              r2=min(r3-1, r4-5)=min(7,3)=3
//              r1=r3-4=4
//              r0=min(r1-2, r2-3)=min(2,0)=0
//   slack = required - arrival:
//              s0=0  s1=2  s2=0  s3=2  s4=0
//   critical path 0->2->4 (all-zero slack).
#include "egg_test.h"
#include "sta.h"

using egg::TimingGraph;
using egg::TimingResult;

static TimingGraph buildHandGraph() {
    TimingGraph g;
    g.numNodes = 5;
    g.numLevels = 3;

    // level of each node and the level partition (nodes already grouped by id).
    g.level = {0, 1, 1, 2, 2};
    g.levelStart = {0, 1, 3, 5};       // L0={0} L1={1,2} L2={3,4}
    g.levelNodes = {0, 1, 2, 3, 4};

    // Fanin CSR, arcs ordered by sink node: 1<-0, 2<-0, 3<-1, 3<-2, 4<-2.
    g.finStart = {0, 0, 1, 2, 4, 5};
    g.finFrom  = {0, 0, 1, 2, 2};
    g.finDelay = {2.0f, 3.0f, 4.0f, 1.0f, 5.0f};

    // Fanout CSR = exact reversal, arcs ordered by source node:
    //   0->1, 0->2, 1->3, 2->3, 2->4.
    g.foutStart = {0, 2, 3, 5, 5, 5};
    g.foutTo    = {1, 2, 3, 3, 4};
    g.foutDelay = {2.0f, 3.0f, 4.0f, 1.0f, 5.0f};
    return g;
}

int main() {
    const TimingGraph g = buildHandGraph();

    // Sanity: the fixture agrees with the header's PI/PO helpers.
    CHECK(g.numArcs() == 5);
    CHECK(g.isPrimaryInput(0));
    CHECK(!g.isPrimaryInput(3));
    CHECK(g.isPrimaryOutput(3));
    CHECK(g.isPrimaryOutput(4));
    CHECK(!g.isPrimaryOutput(0));

    const TimingResult r = egg::staCpu(g);

    // period (critical-path length)
    CHECK(approxEq(r.period, 8.0f));

    // arrival
    CHECK(approxEq(r.arrival[0], 0.0f));
    CHECK(approxEq(r.arrival[1], 2.0f));
    CHECK(approxEq(r.arrival[2], 3.0f));
    CHECK(approxEq(r.arrival[3], 6.0f));
    CHECK(approxEq(r.arrival[4], 8.0f));

    // required
    CHECK(approxEq(r.required[0], 0.0f));
    CHECK(approxEq(r.required[1], 4.0f));
    CHECK(approxEq(r.required[2], 3.0f));
    CHECK(approxEq(r.required[3], 8.0f));
    CHECK(approxEq(r.required[4], 8.0f));

    // slack = required - arrival
    CHECK(approxEq(r.slack[0], 0.0f));
    CHECK(approxEq(r.slack[1], 2.0f));
    CHECK(approxEq(r.slack[2], 0.0f));
    CHECK(approxEq(r.slack[3], 2.0f));
    CHECK(approxEq(r.slack[4], 0.0f));

    std::printf("hand_computed: OK (period=%.1f, critical path 0->2->4)\n", r.period);
    return 0;
}
