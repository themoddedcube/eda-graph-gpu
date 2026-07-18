# Using the `eda-graph-gpu` STA primitive

A practical guide for engineers **outside** this project who want to embed the
level-parallel static timing analysis (STA) primitive in their own EDA tools. Every
type, function, and struct field named here matches the public headers
(`include/sta.h`, `include/timing_graph.h`) exactly — nothing is invented.

Everything lives in namespace `egg`.

---

## 1. What this is / when to use it

This is a **level-parallel, max-plus static timing analysis** primitive: a
block-based (PERT-style) timing propagation over a directed acyclic timing graph.
Given a timing DAG whose nodes are pins and whose arcs carry delays, it computes, for
every node, the **arrival** time (a forward pass: `arrival[v] = max` over the node's
fanin of `arrival[u] + delay(u→v)`, with primary inputs arriving at 0), the
**required** time (a backward pass: `required[v] = min` over the node's fanout of
`required[w] - delay(v→w)`, with primary outputs seeded at the critical-path length),
and the **slack** (`required - arrival`). Nodes sharing a topological **level** are
mutually independent, so each level is one fully data-parallel update — the structure
the GPU path exploits with one kernel launch per level. Use this primitive when you
need graph-based timing propagation on a GPU: no vendor library covers it (cuGraph is
analytics; cuSPARSE SpMV can't be redefined to a `max`/`+` semiring), so this is a
hand-written primitive checked bit-for-bit against a CPU reference.

---

## 2. Data model — building a `TimingGraph`

The graph is struct-of-arrays / CSR so the CPU reference and the CUDA kernel share one
layout with no marshalling gap. The full public struct (`include/timing_graph.h`):

```cpp
struct TimingGraph {
    int numNodes = 0;
    int numLevels = 0;

    std::vector<int>   level;        // [numNodes]      topological level of each node (0 = PI)
    std::vector<int>   levelStart;   // [numLevels + 1] level L = levelNodes[levelStart[L] .. levelStart[L+1])
    std::vector<int>   levelNodes;   // [numNodes]      node ids grouped by level

    // Fanin (incoming) arcs in CSR — drives the forward arrival pass.
    std::vector<int>   finStart;     // [numNodes + 1]  per-node start offset into finFrom/finDelay
    std::vector<int>   finFrom;      // [numArcs]       source node of each fanin arc
    std::vector<float> finDelay;     // [numArcs]       arc delay

    // Fanout (outgoing) arcs in CSR — drives the backward required-time pass.
    std::vector<int>   foutStart;    // [numNodes + 1]  per-node start offset into foutTo/foutDelay
    std::vector<int>   foutTo;       // [numArcs]       sink node of each fanout arc
    std::vector<float> foutDelay;    // [numArcs]       arc delay

    int  numArcs() const;                   // == finFrom.size()
    bool isPrimaryInput(int v)  const;      // finStart[v]  == finStart[v+1]  (empty fanin)
    bool isPrimaryOutput(int v) const;      // foutStart[v] == foutStart[v+1] (empty fanout)
};
```

### Invariants a caller MUST satisfy

The primitive trusts these; it does not validate them for you.

1. **`levelStart`/`levelNodes` are a valid topological partition.** `levelNodes` lists
   every node id exactly once, grouped by level, and `levelStart` has `numLevels + 1`
   entries with `levelStart[0] == 0` and `levelStart[numLevels] == numNodes`. Node
   `v`'s level equals `level[v]`.
2. **Arcs go strictly low→high level.** Every fanin arc `u → v` has
   `level[u] < level[v]`. This is what makes a single forward pass over ascending
   levels correct: when node `v` is processed, all its fanin arrivals are final.
3. **The fanout CSR is the exact reversal of the fanin CSR.** For every fanin arc
   `u → v` with delay `d`, there is a matching fanout arc from `u` to `v` (`foutTo`)
   with the same delay `d` (`foutDelay`), and vice versa. `numArcs()` is derived from
   `finFrom`, so both CSRs must describe the same arc set.
4. **Primary inputs have empty fanin; primary outputs have empty fanout.** A PI has
   `finStart[v] == finStart[v+1]` (arrives at 0); a PO has
   `foutStart[v] == foutStart[v+1]` (seeded with the critical-path period). Use the
   `isPrimaryInput` / `isPrimaryOutput` helpers to test membership.

### Minimal hand-built example (5 nodes, 3 levels)

This is the graph used by the correctness test, verified against the header field
names. Arc labels are delays:

```
        (2)      (4)
    0 ------> 1 ------> 3        primary outputs: 3, 4 (no fanout)
     \                 ^
   (3) \      (1)     /
        > 2 ---------/
          \  (5)
           -------> 4
```

```cpp
#include "sta.h"
using egg::TimingGraph;

TimingGraph buildHandGraph() {
    TimingGraph g;
    g.numNodes  = 5;
    g.numLevels = 3;

    // Level of each node + the level partition (nodes already grouped by id).
    g.level      = {0, 1, 1, 2, 2};
    g.levelStart = {0, 1, 3, 5};        // L0={0}  L1={1,2}  L2={3,4}
    g.levelNodes = {0, 1, 2, 3, 4};

    // Fanin CSR, arcs ordered by sink node: 1<-0, 2<-0, 3<-1, 3<-2, 4<-2.
    g.finStart = {0, 0, 1, 2, 4, 5};
    g.finFrom  = {0, 0, 1, 2, 2};
    g.finDelay = {2.0f, 3.0f, 4.0f, 1.0f, 5.0f};

    // Fanout CSR = exact reversal, arcs ordered by source: 0->1, 0->2, 1->3, 2->3, 2->4.
    g.foutStart = {0, 2, 3, 5, 5, 5};
    g.foutTo    = {1, 2, 3, 3, 4};
    g.foutDelay = {2.0f, 3.0f, 4.0f, 1.0f, 5.0f};
    return g;
}
```

Running `staCpu` on this graph gives `period = 8`, arrivals `{0, 2, 3, 6, 8}`,
requireds `{0, 4, 3, 8, 8}`, and slacks `{0, 2, 0, 2, 0}` — the critical path is
`0 → 2 → 4` (all-zero slack).

### Or generate a synthetic graph

For benchmarking and testing you don't need to hand-build anything. The generator
produces a deterministic layered DAG that already satisfies every invariant above:

```cpp
// numLevels levels of widthPerLevel nodes; each non-source node draws
// 1..maxFaninPerNode fanin arcs from the preceding one or two levels.
TimingGraph generateLayeredDag(int numLevels, int widthPerLevel,
                               int maxFaninPerNode, uint64_t seed);
```

```cpp
egg::TimingGraph g = egg::generateLayeredDag(400, 4000, 4, 0x9E3779B97F4A7C15ULL);
```

> **Current limitation:** the synthetic generator is the only supported graph source
> today. Ingesting a real netlist (a decoupled SoA netlist / Bookshelf + library
> front-end) is on the roadmap (Stage 2), not yet available. Until then, a real tool
> must populate a `TimingGraph` itself, honoring the invariants above.

---

## 3. API reference

All declarations are in `include/sta.h`. The result type:

```cpp
struct TimingResult {
    std::vector<float> arrival;   // latest arrival time at each node
    std::vector<float> required;  // required time at each node
    std::vector<float> slack;     // required - arrival
    float period = 0.0f;          // critical-path length (max arrival over primary outputs)
};
```

### `staCpu` — the CPU reference (and oracle)

```cpp
TimingResult staCpu(const TimingGraph& g);
```

Level-ordered forward (arrival) + backward (required) pass, `O(V + E)`. This is the
reference implementation the GPU path is checked against. It has no device dependency
and runs everywhere.

### `staGpu` — one-shot GPU path with honest CPU fallback

```cpp
TimingResult staGpu(const TimingGraph& g, bool* ranGpu);
```

Runs the level-parallel GPU primitive: one kernel per topological level, all nodes in
a level updated in parallel, over the CSR fanin/fanout arrays. **When no CUDA device is
present it falls back to the CPU reference and sets `*ranGpu = false`** — the same
binary is correct with or without a GPU and never fabricates a GPU result. Always check
`ranGpu` before attributing a result to the GPU.

```cpp
#include "sta.h"

egg::TimingGraph g = egg::generateLayeredDag(400, 4000, 4, 0x9E3779B97F4A7C15ULL);

bool ranGpu = false;
egg::TimingResult r = egg::staGpu(g, &ranGpu);

if (ranGpu) {
    // r came from the GPU primitive.
} else {
    // No CUDA device — r is the CPU reference (honest fallback, not a fake GPU result).
}
printf("critical-path length = %.3f\n", r.period);
```

### Persistent plan — capture once, replay many (`StaGpuPlan`)

Real STA re-evaluates the **same** topology many times (incremental timing inside a
placement / optimization loop). The persistent-plan path uploads the graph once and
captures the entire level sweep (forward + on-device period reduce + backward) into a
CUDA graph a single time, then replays it with one `cudaGraphLaunch` per evaluation —
amortizing the `~2 * numLevels` per-level launch latencies that dominate the naive
path.

```cpp
struct StaGpuPlan;                                   // opaque handle
StaGpuPlan*  staGpuPlanCreate(const TimingGraph& g); // build + capture (one-time)
TimingResult staGpuPlanRun(StaGpuPlan* plan);        // replay (call per evaluation)
void         staGpuPlanDestroy(StaGpuPlan* plan);    // free device resources
```

**Lifecycle:**

1. `staGpuPlanCreate(g)` uploads the graph and captures the sweep. **It returns
   `nullptr` when there is no CUDA device** — callers must check for null and fall back
   to `staCpu`.
2. `staGpuPlanRun(plan)` replays the captured graph and returns a fresh
   `TimingResult`. Call it as many times as you re-evaluate; the capture cost is paid
   only once.
3. `staGpuPlanDestroy(plan)` releases the plan. (Passing a null plan is not required to
   be a no-op — only destroy a plan you actually created.)

The captured math is identical to `staCpu`, so the `maxAbsDiff` oracle still holds.

```cpp
#include "sta.h"

egg::TimingGraph g = egg::generateLayeredDag(400, 4000, 4, /*seed=*/1234);

egg::StaGpuPlan* plan = egg::staGpuPlanCreate(g);   // one-time capture
if (!plan) {
    // No CUDA device — fall back to the CPU reference.
    egg::TimingResult r = egg::staCpu(g);
    // ... use r ...
} else {
    for (int iter = 0; iter < numEvaluations; ++iter) {
        // (real incremental STA would re-time an edited cone here)
        egg::TimingResult r = egg::staGpuPlanRun(plan);   // single cudaGraphLaunch
        // ... consume r.arrival / r.required / r.slack / r.period ...
    }
    egg::staGpuPlanDestroy(plan);
}
```

### `maxAbsDiff` — the equivalence oracle

```cpp
double maxAbsDiff(const TimingResult& a, const TimingResult& b);
```

Order-independent equivalence check: the maximum absolute difference across the
`arrival` and `required` arrays. STA outputs are deterministic value arrays, so this is
the correct comparator — not a path/order diff. Use it to validate any GPU result
against the CPU reference; the project treats a nonzero-beyond-epsilon result as a hard
failure.

```cpp
egg::TimingResult cpu = egg::staCpu(g);
bool ranGpu = false;
egg::TimingResult gpu = egg::staGpu(g, &ranGpu);
if (ranGpu) {
    double d = egg::maxAbsDiff(cpu, gpu);   // expected: 0 (bit-exact) at every size
    assert(d <= 1e-2);
}
```

---

## 4. Build & integrate

Two build systems ship with the repo; both produce the same public API.

### Make

```bash
make          # GPU build via nvcc (runs on GPU if present, else CPU fallback)
make cpu      # CPU-only build via g++ (no CUDA toolchain needed)
./sta 400 4000        # <levels> <nodes-per-level>   (default 400 x 4000)
```

`make` builds with `nvcc -arch=sm_90` by default (override `ARCH`, `NVCC`, `CUDA` on
the make command line). `make cpu` compiles the CPU-only stub with `g++` and needs no
CUDA toolchain.

### CMake

CMake auto-detects a CUDA toolchain (`check_language(CUDA)`): if `nvcc` is present it
compiles the real kernel `src/sta_gpu.cu` (architectures `70;80;90` by default);
otherwise it compiles the CPU-only stub. Either way the public API is identical and
honest.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sta -j
```

### Linking into your own project

CMake bundles the CPU reference, generator, and the chosen GPU source into a single
static library target, **`egg_sta`**, with `include/` exposed as a public include
directory. Link against it:

```cmake
target_link_libraries(your_tool PRIVATE egg_sta)
```

Then include the public header and call the API:

```cpp
#include "sta.h"          // pulls in timing_graph.h; everything is in namespace egg
```

If you integrate outside CMake, add the repo's `include/` to your include path and link
the compiled `egg_sta` static library. Core STA is pure C++17 (no CUDA required to
build or run the CPU path).

---

## 5. Measured performance

The numbers below are reproduced from the repo `README.md`, which is the authoritative
source. **Measured on an H100 PCIe (CUDA 12.8, min-of-5)**; every GPU number is
measured on-device and **bit-exact** against the CPU reference (`maxAbsDiff == 0` at
every size). `warm` is the per-launch level sweep with the CUDA context already live;
`graph-replay` is the persistent-plan CUDA-graph path. Speedups are vs a single-thread
CPU reference.

| nodes | arcs | CPU | GPU warm (per-launch) | GPU graph-replay | oracle |
|---:|---:|---:|---:|---:|:--:|
| 65,536 | 162,565 | 1.27 ms | 1.72 ms · 0.74× | **0.82 ms · 1.54×** | MATCH |
| 262,144 | 652,834 | 5.10 ms | 3.79 ms · 1.35× | **1.81 ms · 2.82×** | MATCH |
| 1,600,000 | 3,991,836 | 32.9 ms | 10.4 ms · 3.18× | **5.74 ms · 5.73×** | MATCH |
| 6,400,000 | 15,980,582 | 132 ms | 33.0 ms · 3.99× | **16.1 ms · 8.18×** | MATCH |

Read it honestly (per the README): the *first* GPU call pays a one-time CUDA context
init (~210–310 ms), reported separately and never hidden; the naive per-launch sweep is
launch-bound and **loses on small graphs** (0.74× at 65k); the CUDA-graph plan collapses
the `~2 * numLevels` per-level launches into one replay, buying 1.8–2.1× over per-launch
and scaling to 8.2× at 6.4M nodes.

### Regenerate the table

```bash
python3 bench/bench.py
```

`bench.py` configures + builds via CMake, sweeps graph sizes, and writes
`bench/results.csv` plus a printed table. It records what the binary actually measured:
on a GPU box it parses the on-device warm and graph-replay times and speedups; on a box
with no GPU it records `mode = "cpu-only(fallback)"` with empty GPU columns and never
invents a GPU time. Useful flags: `--no-build` (reuse a build), `--reps N` (min-of-N
inside the binary), `--sizes 128x512,400x4000`, `--csv out.csv`.

---

## 6. The honesty contract & limitations

This primitive is built to a strict honesty contract — worth understanding before you
depend on it or cite its numbers:

- **Bit-exact oracle.** The GPU result is compared to the CPU reference with the
  order-independent `maxAbsDiff` comparator over the arrival/required arrays. Reported
  results are `MATCH` only at `maxAbsDiff == 0`; a mismatch is a hard failure, never a
  relaxed gate.
- **No device → honest CPU fallback.** With no CUDA device, `staGpu` returns the CPU
  reference and sets `*ranGpu = false`, and `staGpuPlanCreate` returns `nullptr`. The
  same binary is correct everywhere and **never fabricates a GPU result or speedup** —
  always branch on `ranGpu` / a null plan.
- **Measured ≠ modeled.** Every performance number reported by the binary and by
  `bench.py` is *measured* on the host it ran on. The optimization roadmap in
  `docs/research/` also carries first-order *estimates* and *paper-measured* figures
  from other workloads; those are labeled as such and are not predictions for this
  kernel.
- **Graph source is the synthetic generator (today).** `generateLayeredDag` is the only
  supported way to obtain a graph. Real-netlist ingestion is on the roadmap (Stage 2),
  not yet implemented; a real tool must build its own `TimingGraph` honoring the Section
  2 invariants.
- **Single precision.** Arrival, required, slack, delays, and period are all `float`.
  The model uses lumped per-arc delays (no current-source / waveform delay model), and
  the traversal is block-based (PERT), not path-based.

---

*Source of truth: `include/sta.h`, `include/timing_graph.h`, `src/sta_cpu.cpp`,
`src/main.cpp`, and the repo `README.md`. This page paraphrases them; where they
disagree with this page, they win.*
