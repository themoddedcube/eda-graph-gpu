# A Level-Parallel GPU Primitive for Static Timing Analysis: An Honest Build-vs-Buy Account

*eda-graph-gpu — engineering paper drawn from the project lab notebook*

Hardware for every result labeled **measured**, unless stated otherwise: NVIDIA H100
PCIe (80 GB), driver 570.124.06, CUDA 12.8; CPU reference compiled `-O2`; the 24-core
CPU baseline (`staCpuParallel`) is the same level decomposition via OpenMP, bit-identical
to the serial reference. Every quantitative claim below traces to a numbered entry
(E1–E10) in `docs/lab-notebook.md`; the entry is cited inline (e.g. "(E6)"). Numbers
that are *estimated* or *paper-measured on another workload* are labeled as such and are
never presented as our measurements.

---

## Abstract

Static timing analysis (STA) is one of the runtime-dominating graph workloads in EDA,
yet it has no drop-in GPU library: cuGraph is an analytics package (BFS, PageRank,
connected components) and the CUDA math libraries (cuFFT/cuBLAS/cuSPARSE) do not express
max-plus timing propagation over a topological DAG. We build the missing primitive — a
level-parallel, block-based (PERT-style) STA kernel — and hold it to a strict honesty
contract: every GPU result is checked bit-for-bit against a CPU reference by an
order-independent oracle, and every performance number is measured on-device or it is not
reported.

Our central finding reproduces, on our own kernel, the thesis that a naive GPU offload is
often a net loss. Against a *fair all-core* CPU baseline (24 threads, the same level
decomposition), the naive per-launch level sweep merely ties the CPU (0.85–1.31×) and
loses outright on small graphs (E2, E6). The win comes from an engineering change, not the
raw offload: capturing the entire forward + on-device period-reduce + backward chain into
a **CUDA graph** and replaying it collapses the ~2·numLevels per-level launches, buying
1.8–2.1× over the per-launch path and yielding an honest **1.7–2.6×** over 24 cores (E4,
E6). Correctness is validated bit-exact on all 11 ISCAS-85 circuits (E7) and, against a
double-precision ground truth, the fp32 result is accurate to ~1 ppm relative on arrival
even at 1200 levels (E10). We also demonstrate the "buy" half of the toolbox — routing
BFS/WCC/SSSP to cuGraph, oracle-checked, while topological sort and DFS (no cuGraph
primitive) are built (E9). We keep the limitations in view: the synthetic generator is
more regular than real circuits, the real benchmarks are small, all numbers are on a
single GPU, and the corner-sweep number honestly includes its host-to-device (H2D)
transfer cost.

---

## 1. Introduction and Motivation

EDA is saturated with graph algorithms, but the ones that dominate its runtime — static
timing analysis (STA), global/detailed routing, logic-synthesis DAG traversal — have no
drop-in GPU library. cuGraph provides analytics primitives (PageRank, community
detection, centrality); the CUDA Toolkit provides cuFFT/cuBLAS/cuSPARSE for numerical
math, not timing propagation. For these workloads there is nothing to route *to*: you have
to build the primitive. That is what this project is — hand-written, CPU-oracle-checked
GPU primitives for the uncovered EDA graph workloads, decided per workload by a single
build-vs-buy question: *does a strong vendor primitive already exist?*

| Workload | Right answer |
|---|---|
| FFT (placement density), dense LA, sparse solve | **Use the library** — cuFFT / cuBLAS / cuSPARSE. Faster and verified; do not reinvent. |
| Netlist connectivity / reachability / cones | **Use cuGraph** — BFS / weakly-connected components genuinely fit. |
| **STA timing propagation, routing wavefronts** | **Build our own** — no vendor primitive exists. *This work.* |

**The thesis under test — the naive GPU port is often a net loss.** A level sweep looks
embarrassingly parallel, so the tempting move is to launch one kernel per topological
level and declare victory over a single CPU thread. We deliberately did *not* stop there.
When we measured the naive per-launch primitive against a *fair, all-core* CPU baseline,
it merely tied the CPU and lost on small graphs (E2, E6). The apparent 5–8× "GPU win" is
an artifact of comparing against one core; it does not survive an honest baseline. The
result that *does* survive comes from removing the structural cost the measurement
exposed — kernel-launch overhead — via CUDA-graph capture/replay.

**Contributions.**

- A **level-parallel, block-based STA GPU primitive** (max-plus forward arrival, min-plus
  backward required, fused slack) with a CPU reference that doubles as an
  order-independent equivalence oracle; bit-exact at every measured size (E1, E2).
- A **measured demonstration of the naive-offload-is-a-net-loss thesis** on our own kernel:
  against a fair 24-core CPU baseline the naive per-launch sweep only ties the CPU
  (0.85–1.31×) and loses on small graphs (E2, E6).
- Two measured optimizations that earn the GPU its keep: **on-device period reduction +
  fused slack** removing the mid-pipeline host round-trip (E3), and **CUDA-graph
  capture/replay** buying 1.8–2.1× over per-launch and an honest 1.7–2.6× over 24 cores
  (E4, E6).
- A **changing-input replay path** (`staGpuPlanUpdateDelays`) that turns capture/replay
  into an honest multi-corner / Monte-Carlo re-evaluation, with the per-corner H2D upload
  measured, not hidden (E8).
- A **double-precision ground truth** that bounds the fp32 error (~1 ppm relative arrival,
  ~2.5e-3 near-zero slack even at 1200 levels), replacing a self-referential fp32-vs-fp32
  "bit-exact" claim with a real error bound (E10).
- Correctness on **real netlists**: bit-exact STA on all 11 ISCAS-85 circuits, plus a
  real-vs-synthetic topology profile that names the generator's realism gap (E7).
- The **"buy" complement**: BFS/WCC/SSSP routed to cuGraph and oracle-checked on all 11
  ISCAS-85 circuits; topological sort and DFS have no cuGraph primitive, so we build them
  (E9).

---

## 2. Background — STA as max-plus propagation over a DAG

A timing graph is a directed acyclic graph (DAG) whose nodes are pins and whose arcs carry
delays. Block-based (PERT-style, graph-based) STA computes, for every node, three
quantities.

**Forward pass — arrival.** The latest arrival time at a node is the max over its fanin of
the driver's arrival plus the arc delay:

```
arrival[v] = max over u in fanin(v) of ( arrival[u] + delay(u -> v) )
```

Primary inputs (empty fanin) arrive at 0. This is a **max-plus** semiring reduction: `max`
in place of `+`, `+` in place of `×`. Structurally the forward pass is a max-plus sparse
matrix–vector product — for each row (node) it reduces over that row's nonzeros (fanin
arcs) — which is why the CSR-SpMV load-balancing literature is relevant to it.

**Backward pass — required.** The required time is the min over the node's fanout of the
sink's required time minus the arc delay:

```
required[v] = min over w in fanout(v) of ( required[w] - delay(v -> w) )
```

Primary outputs (empty fanout) are seeded with the critical-path length (the **period** =
max arrival over the primary outputs).

**Slack.** `slack[v] = required[v] - arrival[v]`. The critical path is the all-zero-slack
path; slack near zero is precisely the timing-critical region a designer cares about (and,
being a subtraction, the region where floating-point cancellation can bite — see §6.5).

**Level parallelism.** Assign each node a topological **level** (longest-path-from-source
depth). All nodes at the same level are mutually independent: none is in another's fanin.
So each level is one fully data-parallel update, and the whole forward pass is a serial
sweep over ascending levels with full parallelism *within* each level; the backward pass
mirrors it over descending levels. This is the textbook GPU level-sweep, and it is exactly
what the primitive exploits — one kernel launch per level, one thread per node, over a CSR
fanin/fanout layout.

---

## 3. Design

### 3.1 CPU reference and the equivalence oracle (E1)

The primitive is built on a struct-of-arrays / CSR timing graph (`include/timing_graph.h`)
so the CPU reference and the CUDA kernel share one memory layout with no marshalling gap.
On top of it sit four pieces (E1):

- **A level-ordered CPU reference** (`src/sta_cpu.cpp` `staCpu`): forward arrival = max over
  fanin of `arrival + delay`; backward required = min over fanout of `required − delay`;
  slack = required − arrival. `O(V + E)`, no device dependency, runs everywhere. This
  reference is the correctness oracle.
- **An order-independent equivalence comparator** (`maxAbsDiff`): the maximum absolute
  difference across the `arrival` and `required` arrays. STA outputs are deterministic
  value arrays, so this — not a path/order diff — is the correct comparator. The project
  treats any result beyond float epsilon as a hard failure (exit 1); a mismatch is never a
  relaxed gate.
- **A deterministic synthetic layered-DAG generator** (`generateLayeredDag`) that satisfies
  every graph invariant by construction, for benchmarking and testing.

**Correctness anchor.** A hand-computed 5-node DAG, solved on paper, matches `staCpu`
exactly (`tests/test_hand_computed.cpp`: period 8, critical path 0→2→4), and `ctest` is
green (E1).

### 3.2 The level-parallel GPU primitive (E2)

`src/sta_gpu.cu` implements the level sweep directly: **one kernel launch per topological
level**, **one thread per node**, reducing over that node's fanin/fanout in CSR. Serial
across levels, parallel within. This is the naive, faithful port — deliberately kept as
the baseline against which the optimizations are measured.

The primitive carries an **honest CPU fallback**: with no CUDA device, `staGpu` returns the
CPU reference and sets `*ranGpu = false` (and `staGpuPlanCreate` returns `nullptr`). The
same binary is correct with or without a GPU and never fabricates a GPU result or speedup.

### 3.3 Iteration 1 — on-device period reduction + fused slack (E3)

The first optimization removes a mid-pipeline host round-trip. The naive path synced after
the forward sweep, copied the whole `arrival` array to the host, and reduced the period in
a host loop before relaunching the backward sweep. Iteration 1 replaces that with a device
path: `poMaskKernel` masks non-primary-outputs to −inf and `cub::DeviceReduce::Max`
produces the period on-device (this is a "buy" — the period reduction is a standard CUB
primitive, not hand-rolled), and slack is **fused** into `requiredKernel`. One
`cudaDeviceSynchronize` plus one device-to-host copy of the results now covers the whole
pipeline. The math is identical (same max/min/subtract), so equivalence holds by
construction and is confirmed on-device (oracle MATCH at every size). Chosen first because
it is the lowest-risk change and the **enabler** for iteration 2: with no host interruption
between the two sweeps, the entire pipeline can sit in one CUDA graph (E3).

### 3.4 Iteration 2 — CUDA-graph capture/replay (E4)

E2's measurement showed the naive primitive is **launch-bound**: the per-level compute is
tiny, so the ~2·numLevels (≈800–1600) per-level kernel launches dominate. Iteration 2 adds
a persistent plan (`StaGpuPlan` / `staGpuPlanCreate` / `staGpuPlanRun` / `staGpuPlanDestroy`)
that **stream-captures the entire forward + period-reduce + backward chain into a CUDA graph
once**, then replays it with a single `cudaGraphLaunch`. The kernel bodies are unchanged; the
only behavioral difference is *when* the launches hit the driver. Graph *capture* is a
one-time cost (measured 1.3–16.7 ms), reported separately and amortized over repeated
evaluations (E4).

### 3.5 Changing-input replay for multi-corner / Monte-Carlo re-evaluation (E8)

A static CUDA graph is topology-static, so replaying it must model a workload where the
*topology* is fixed but the *inputs* change. The honest such workload is **multi-corner /
Monte-Carlo STA**: one fixed topology re-evaluated under many different arc-delay sets.
`staGpuPlanUpdateDelays(plan, finDelay, foutDelay)` `cudaMemcpy`s a new delay set into the
plan's device buffers; because the captured graph references the same pointers, **no
re-capture** is needed — the next replay simply uses the new delays. A wrong-size update is
rejected and does not corrupt the plan. `withFinDelay` (`src/circuit.cpp`) builds the
matching CPU oracle for a corner, so each corner's GPU result can be checked against its own
CPU solution (E8).

### 3.6 The double-precision ground truth (E10)

`staCpuDouble` (`src/sta_double.cpp`) is the same STA in double precision — the truth for the
exact float inputs — and `fp32Error` bounds the fp32 result's error, including slack in the
cancellation-prone near-zero region. This replaces a self-referential fp32-vs-fp32 check
with an error bound against ground truth (E10; results in §6.5).

---

## 4. Methodology — the honesty contract

Every result in this paper is produced under a fixed set of rules, inherited from the parent
CUDAadvisor project's lab-notebook discipline.

- **Measured ≠ estimated.** A number that came out of a real on-device run is labeled
  *measured*; a modeled number is labeled *estimated*; a number a cited paper measured on a
  different workload is labeled *paper-measured*. One is never dressed as another. The
  optimization roadmap's pre-measurement estimates are kept, labeled as estimates, even where
  the measured effect diverged (§6.2).
- **The oracle.** The GPU result is compared to the CPU reference with the order-independent
  `maxAbsDiff` comparator over the arrival/required arrays. `MATCH` means `maxAbsDiff == 0`
  (bit-exact); anything beyond float epsilon is a hard failure. No validation gate is relaxed
  to make a result look like it passed.
- **A fair, all-core CPU baseline.** The headline speedup is against **all 24 CPU cores**
  (`staCpuParallel` — the same level decomposition via OpenMP, asserted bit-identical to the
  serial reference in `tests/test_oracle_consistency.cpp`), *not* one thread (E6). The
  single-core number is retained only as a secondary datapoint. The 1-core-strawman headline
  is explicitly retired.
- **No fabricated GPU numbers.** With no CUDA device the primitive falls back to the CPU
  reference and says so; `bench.py` records `mode = "cpu-only(fallback)"` with empty GPU
  columns and never invents a GPU time. On a CPU-only box the GPU cross-checks skip honestly
  (still exit 0).
- **Cold vs warm separation.** The first GPU call pays a one-time CUDA context init
  (~210–310 ms); it is reported as a separate **cold** number, never folded into the **warm**
  steady-state time that a repeated-evaluation loop actually sees (E2).
- **Reproducible.** Each measured entry ships with the exact command that produced it and a
  correctness check (the `MATCH` oracle line).

Two methodology traps were caught and fixed while building the fair baseline, and are logged
in the open (E6): a **silent single-threading** bug where the OpenMP thread count followed an
ambient `OMP_NUM_THREADS` / inherited affinity mask (a subprocess baseline silently ran on 1
thread while claiming "24-core"), fixed by setting the count explicitly to
`omp_get_num_procs()`; and a **parser bug** where `bench.py`'s `(\d+)-core` regex matched the
1-core line first and mislabeled it the multi-core baseline, fixed to select the max-core
line. "Measured" includes measuring the harness.

---

## 5. Evaluation

All results are measured on the H100 configuration named in the header, with the CPU baseline
being all 24 cores unless a single-core column is explicitly shown.

### 5.1 The naive per-launch primitive is a net loss vs a fair baseline (E2, E6)

The naive per-launch level sweep, timed warm (min-of-5), against the single-thread reference
(E2):

| nodes | arcs | CPU (1-core) | GPU cold (incl. 1-time init) | GPU warm (per-launch) | oracle |
|---:|---:|---:|---:|---:|:--:|
| 65,536 | 162,565 | 1.27 ms | 289 ms | 1.72 ms · **0.74×** | MATCH (0.0) |
| 262,144 | 652,834 | 5.10 ms | 215 ms | 3.79 ms · **1.35×** | MATCH (0.0) |
| 1,600,000 | 3,991,836 | 32.9 ms | 222 ms | 10.4 ms · **3.18×** | MATCH (0.0) |
| 6,400,000 | 15,980,582 | 132 ms | 309 ms | 33.0 ms · **3.99×** | MATCH (0.0) |

*Measured, min-of-5, H100 (E2).* The cold floor (~210–310 ms) is nearly constant across
problem size — it is context init, not compute — and warm the naive sweep **loses on small
graphs** (0.74× at 65k), crossing over CPU only near 262k nodes (E2). This is the
naive-offload-is-a-net-loss thesis reproduced on our own primitive. Even the 3.99× at 6.4M
is against a *single* core, which the next table shows to be a strawman.

Against the **fair 24-core baseline** (min-of-15), the naive per-launch GPU merely ties the
CPU (E6):

| nodes | CPU 1-core | CPU 24-core | GPU per-launch | GPU graph-replay | replay vs 24-core |
|---:|---:|---:|---:|---:|---:|
| 262,144 | 5.08 ms | 3.53 ms | 3.95 ms | 1.82 ms | **1.9×** |
| 1,600,000 | 32.8 ms | 16.2 ms | 12.4 ms | 6.19 ms | **2.6×** |
| 6,400,000 | 131 ms | 27.6 ms | 32.5 ms | 15.9 ms | **1.7×** |

*Measured, min-of-15, H100 vs all 24 cores (E6).* The per-launch GPU is 0.85–1.31× vs the
24-core CPU — it does not earn its keep on the raw offload. The headline number changed by
~4× once the baseline was made fair (from ~8× vs 1 core to ~1.7–2.6× vs 24 cores). Per
append-only discipline, the earlier 1-core entries (E2, E4) are kept in place, uncorrected,
and this entry (E6) corrects their framing.

### 5.2 CUDA-graph replay is what earns the GPU its keep (E4, E6)

Adding CUDA-graph capture/replay (min-of-5, vs the 1-core reference, E4):

| nodes | CPU (1-core) | GPU warm (per-launch) | GPU graph-replay | replay vs per-launch | oracle |
|---:|---:|---:|---:|---:|:--:|
| 65,536 | 1.27 ms | 1.72 ms · 0.74× | **0.82 ms · 1.54×** | **2.09×** | MATCH (0.0) |
| 262,144 | 5.10 ms | 3.79 ms · 1.35× | **1.81 ms · 2.82×** | **2.09×** | MATCH (0.0) |
| 1,600,000 | 32.9 ms | 10.4 ms · 3.18× | **5.74 ms · 5.73×** | **1.80×** | MATCH (0.0) |
| 6,400,000 | 132 ms | 33.0 ms · 3.99× | **16.1 ms · 8.18×** | **2.05×** | MATCH (0.0) |

*Measured, min-of-5, H100 (E4).* Replay buys **1.8–2.1× over the per-launch path**, pushes
the CPU-crossover **below 65k** (0.74× → 1.54×), and scales to 8.18× vs a single core at 6.4M
— bit-exact everywhere. This directly confirms E2's diagnosis that the primitive was
launch-bound. Read against the fair 24-core baseline (§5.1, E6), replay delivers the honest
**1.7–2.6×**, and it is the CUDA-graph optimization — not the raw offload — that makes the
GPU worth using.

### 5.3 Correctness on real ISCAS-85 netlists (E7)

Correctness is not shown only on the synthetic generator. A general DAG **levelizer**
(`src/circuit.cpp` `levelize`: longest-path-from-source levels, cycle rejection) and an
ISCAS-85 `.bench` reader (`readBench`) ingest all 11 public-domain ISCAS-85 circuits. STA is
**bit-exact (oracle MATCH) on all 11 real circuits**, CPU and GPU (`tests/test_bench_circuits.cpp`,
6/6 ctest green); c17 is checked against a hand-derived result (11 nodes, 12 arcs, 4 levels,
5 PI, 2 PO, period 6) (E7).

The topology profiler (`./build/profile`) puts real circuits beside the generator (E7):

| graph | nodes | levels | meanW | maxW | meanFanin | maxFanin | maxFanout |
|---|---:|---:|---:|---:|---:|---:|---:|
| c432 | 196 | 18 | 10.9 | 36 | 1.71 | 9 | 9 |
| c2670 | 1426 | 33 | 43.2 | 233 | 1.46 | 5 | 233 |
| c7552 | 3719 | 44 | 84.5 | 355 | 1.65 | 5 | 207 |
| syn-mid | 65,536 | 128 | 512 | 512 | 2.48 | 4 | 11 |
| syn-big | 1,600,000 | 400 | 4000 | 4000 | 2.49 | 4 | 14 |

*Measured (E7).* Three honest findings, in both directions: (1) correctness generalizes to
real, irregular topology — it is not a synthetic artifact; (2) the real *available*
benchmarks are tiny (≤3.7k gates) with narrow irregular levels, so on them the GPU is
launch-bound and the **CPU is the right tool** — the GPU regime is the large-scale graph
(10⁵–10⁷ nodes) modern designs reach; (3) the generator is **not fully faithful** — its
levels are perfectly uniform (meanW == maxW) and its fanin is banded ≤4, whereas real
circuits have irregular widths and high-fanout hubs (maxFanout up to ~300) and higher maxFanin.
So the large-scale synthetic numbers (§5.1–5.2) live on a *more regular* graph than reality;
this is a real limitation (§7), tracked toward a realistic large generator.

### 5.4 Multi-corner re-evaluation with changing delays (E8)

The corner sweep (H100, 1.6M-node graph, K=16 delay sets), where each corner re-evaluates the
fixed topology under a different set of arc delays (E8):

| configuration | time | note |
|---|---:|---|
| CPU 24-core (re-solve each corner) | 208.5 ms | fair all-core baseline |
| GPU plan (updateDelays + replay per corner) | 171.4 ms | **1.22×**, results vary (\|c0−clast\|=153) |

*Measured, H100 (E8).* Correctness on changing input is proven, not assumed: for each of 6
non-uniform corners (every arc scaled by a random factor in [0.5,1.5]), the GPU result (a)
matches `staCpu` on that corner's delays (`maxAbsDiff ≤ 1e-2`) and (b) differs from the base
corner (`maxAbsDiff > 0.5`) — so replay is provably not re-running one answer
(`tests/test_corner_replay.cpp`, 7/7 ctest green). The speedup is only ~1.2× here, *not* the
1.7× of a single evaluation, because the sweep **includes the per-corner H2D upload of the
delay arrays** (~2·numArcs floats × K) — a real cost deliberately timed, and exactly the H2D
transfer a reviewer's fairness point cares about. The win is modest but genuine; on-device
corner generation to remove that upload is future work (§10).

### 5.5 fp32 error bounded against a double-precision truth (E10)

`maxAbsDiff == 0` only shows the GPU matches the *equally-rounded* fp32 CPU reference. Against
`staCpuDouble` — the double-precision truth for these exact float inputs — the fp32 error, on
deep graphs where it is worst (E10):

| levels | period | rel arrival err | max abs slack err | worst near-zero slack err |
|---:|---:|---:|---:|---:|
| 400 | 1453 | 7.3e-07 | 1.4e-03 | 2.0e-04 |
| 800 | 2911 | 1.3e-06 | 4.5e-03 | 1.2e-03 |
| 1200 | 4324 | 1.2e-06 | 5.2e-03 | 2.5e-03 |

*Measured (`tests/test_precision.cpp`, E10).* Even at **1200 levels** of accumulation the fp32
result is accurate to **~1 ppm** relative on arrival, and slack in the cancellation-prone
near-zero region is good to **~2.5e-3** absolute — orders of magnitude below any real timing
margin. So the fp32 primitive is trustworthy against ground truth, not merely self-consistent;
the GPU result (also fp32) is held to the same bound. The test asserts explicit bounds
(relArrival < 1e-4, near-zero slack < 1e-1) so a regression fails CI (E10).

### 5.6 Edge-case and property test suite (E5)

Beyond the size sweep, an edge/base-case and property suite (`tests/test_edge_cases.cpp`,
E5) exercises degenerate graphs with the arithmetic worked out in comments — a single node
(PI ≡ PO), a single all-PI/PO level, a pure chain, **zero-delay arcs** (which exercise the
PO-masked period reduction, correct even when an internal node ties a PO), and a
**high-fanin star** — plus structural property checks across seeds/sizes (PI arrival 0; PO
required = period; slack = required − arrival; arrival independently recomputed as max over
fanin; period = max PO arrival), generator determinism, and a GPU cross-check of both `staGpu`
and the CUDA-graph plan on every case when a device is present. `ctest` is 5/5 green on the
H100, and on a CPU-only box the GPU branches skip honestly (E5).

---

## 6. The cuGraph build-vs-buy routing — the "buy" half (E9)

The STA primitive is the **build** side of the toolbox. The complement is the standard EDA
graph traversals routed to **cuGraph** where it has a real parallel primitive, and kept on our
own reference where it does not (`graph_routing/`). RAPIDS cuGraph 26.06 is installed into a
gitignored `.venv-rapids`; it imports in ~1.3 s and runs on the H100 (E9).

**The mapping (honest build-vs-buy) (E9):**

| graph op (EDA use) | cuGraph primitive | decision |
|---|---|---|
| BFS reachability / logic cone | `cugraph.bfs` | **buy** — validated |
| Connected components / netlist partition | `cugraph.weakly_connected_components` | **buy** — validated |
| Shortest path (weighted) | `cugraph.sssp` | **buy** — validated |
| Centrality / influence | `cugraph.pagerank` | buy (routed; not oracle-checked here) |
| Strongly-connected components | `cugraph.strongly_connected_components` | buy (routed) |
| **Topological sort / levelization** | **— none —** | **build** → our levelizer |
| **DFS ordering** | **— none —** | **build** → CPU reference |

The honest finding: cuGraph is an *analytics* library — it provides the parallel-friendly
primitives (BFS, WCC/SCC, SSSP, PageRank) but has **no topological sort and no DFS** (both
inherently sequential). Topological sort is exactly the levelization the STA primitive needs;
neither it nor DFS has a cuGraph primitive to route to, so we build them. That asymmetry —
buy where a strong primitive exists, build where it does not — is the whole point of a toolbox
over a hammer.

**Validation (measured, E9).** Each routed primitive runs on the GPU and is checked
**bit-for-bit against a pure-Python oracle** (`graphkit.py`) on a synthetic graph and **all 11
ISCAS-85 circuits**. Every case matches ("ROUTING OK — every cuGraph result matched the
oracle"); e.g. `c2670: bfs MATCH (reach=5), wcc MATCH (80 comps), sssp MATCH` and
`c7552: bfs MATCH (reach=8), wcc MATCH (5 comps), sssp MATCH`. Without cuGraph the reference
oracle still runs and routing reports "wired, GPU validation skipped" — never a fake pass.

Two real-netlist **convention** findings surfaced (both fixed, neither a cuGraph defect) (E9):
**WCC on isolated nodes** — real circuits have unconnected primary inputs that the union-find
oracle counts as their own components, but cuGraph never sees a node absent from the edgelist
(c2670 exposed it: oracle 80 vs cuGraph-lumped 5); the adapter gives each absent node its own
singleton component, and 80 == 80. **SSSP on parallel arcs** — a gate with a repeated input
yields parallel arcs whose arbitrary per-index weights made the min-path ambiguous (c3540
mismatch); SSSP is run on the deduped simple graph so both sides see identical weights.

---

## 7. Limitations and threats to validity

We keep these in view rather than hiding them; the project's red-team review
(`docs/red-team-review.md`) enumerates them and tracks their status.

- **Synthetic-generator realism (partially addressed).** The large-scale numbers (§5.1–5.2)
  are on `generateLayeredDag`, whose levels are perfectly uniform (meanW == maxW), whose fanin
  is banded ≤4, and whose arcs draw only from the previous one or two levels — suppressing the
  irregular widths, high-fanout hubs (real maxFanout up to ~300), long-range arcs, and
  reconvergence of real circuits (E7). The generator is thus the most GPU-favorable topology;
  the win is demonstrated on a *more regular* graph than reality. Correctness (not speed) is
  shown to transfer to real irregular topology on all 11 ISCAS-85 circuits (E7). A realistic
  large generator (skewed fanout, irregular widths, reconvergence) is the top open item.
- **Small real benchmarks.** The available real netlists (ISCAS-85) are tiny (≤3.7k gates),
  where the GPU is launch-bound and the CPU is the right tool (E7); they validate correctness,
  not the large-scale speedup. The speedup regime (10⁵–10⁷ nodes) is exercised only on the
  synthetic graph. A large industrial design is not yet in hand.
- **Single GPU, single architecture.** Every measured number is one H100 PCIe at sm_90. The
  launch-bound / bandwidth-bound behavior is device-specific, so the CPU-crossover point
  (~65k nodes) and the peak speedup will shift on other hardware (A100, L40S, consumer). No
  second-architecture sweep exists.
- **H2D cost in the corner sweep.** The multi-corner number (§5.4, 1.22×) is lower than a
  single evaluation's 1.7× precisely because it **honestly includes** the per-corner H2D delay
  upload (E8). It is a real cost, deliberately timed; on-device corner generation would remove
  it but is not yet built.
- **Noisy shared-box baseline.** The 24-core CPU baseline is memory-bandwidth-bound and noisy
  on the shared cloud host: the min-of-15 `replay vs 24-core` ratio moved ±0.4× between runs
  with CPU contention (a stray 96%-CPU neighbor process was visible during one sweep). The GPU
  times are stable; the CPU-relative ratios are approximate, not to three digits. A quiet,
  named, isolated host is a next step (E6).
- **Not yet compared head-to-head with prior GPU-STA.** Warp-STAR and ICCAD'20 are cited (§8)
  but not run on shared benchmarks on the same host, so this primitive cannot yet be placed on
  the same absolute axis as the closest prior art.
- **Model scope.** Delays are lumped per-arc `float`; the traversal is block-based (PERT), not
  path-based, and there is no current-source / waveform delay model. Single precision is bounded
  against fp64 (§5.5) but remains single precision.

---

## 8. Related work

**GPU block-based STA — the directly comparable prior art.** Guo, Huang, and Lin,
"GPU-Accelerated Static Timing Analysis" (ICCAD 2020), build GPU-efficient data structures and
kernels for levelization, delay calculation, and graph update inside a task-based CPU-GPU
heterogeneous framework, and report (paper-measured) up to **3.69×** on a 1.6M-gate design vs
OpenTimer on 40 CPUs — the closest published analogue to our level sweep.
https://dl.acm.org/doi/10.1145/3400302.3415631 (PDF:
https://tsung-wei-huang.github.io/papers/iccad20-gpusta.pdf). **Warp-STAR**
("High-performance, Differentiable GPU-Accelerated STA through Warp-oriented Parallel
Orchestration") directly attacks the intra-warp load imbalance our thread-per-node kernel has:
prior GPU STA mapped one net to one thread and diverged on unequal pin counts, whereas Warp-STAR
processes a net's pins by the unit of a warp with a shared-memory tree reduction, still launching
one kernel per topological level; it reports (paper-measured) **2.4× over the prior GPU-Timer
and 162× over OpenTimer** (superblue1: 5095 ms CPU → 57 ms GPU-Timer → 25 ms Warp-STAR), and
notes its own negative result that a cleverer "compressed" balancer did *not* beat the simple
warp-per-net scheme. https://arxiv.org/html/2603.28381. A related heterogeneous-parallelism line,
"Accelerating STA using CPU-GPU Heterogeneous Parallelism," reinforces the overlapped
compute/data-movement idea. https://www.researchgate.net/publication/371589918.

**GPU path-based and current-source-model STA (future stages).** Guo, Huang, Lin, and Wong,
"GPU-accelerated Path-Based Timing Analysis" (DAC 2021), report (paper-measured) 543× on leon2,
172× on leon3mp, and 304× on netcard vs a CPU baseline.
https://dl.acm.org/doi/10.1109/DAC18074.2021.9586316 (PDF:
https://yibolin.com/publications/papers/TIMER_TopK_DAC2021_Guo.pdf); its journal extension is
"A GPU-Accelerated Framework for Path-Based Timing Analysis" (TCAD 2023),
https://dl.acm.org/doi/10.1109/TCAD.2023.3272274. **GCS-Timer** (DAC 2024) moves beyond lumped
arc delays to current-source-model waveform propagation on GPU,
https://dl.acm.org/doi/10.1145/3649329.3655983, and **INSTA** (DAC 2025, NVIDIA) pursues
differentiable/statistical STA on GPU,
https://research.nvidia.com/labs/electronic-design-automation/papers/yichen_INSTA_dac25.pdf.

**CPU incremental STA.** OpenTimer v2 is built around a task-based incremental engine that
re-times only the affected fan-in/fan-out cone on a netlist edit — the yardstick CPU engine for
this workload.

**Transferable kernel technique base.** Our forward pass is structurally a max-plus SpMV, so the
CSR-SpMV load-balancing literature applies: Anzt et al., "Load-balancing Sparse Matrix Vector
Product Kernels on GPUs" (ACM TOPC 2020), https://dl.acm.org/doi/fullHtml/10.1145/3380930;
LightSpMV's dynamic warp/vector assignment, https://www.researchgate.net/publication/276411706;
and Bell & Garland, "Implementing SpMV on Throughput-Oriented Processors" (SC09), the origin of
CSR-vector / hybrid formats, https://www.nvidia.com/docs/io/77944/sc09-spmv-throughput.pdf.
Warp-level reductions (`cub::WarpReduce` / `__shfl_down_sync`) are the natural tool for a
warp-cooperative fanin reduction. For launch-overhead amortization — our biggest structural cost
— a kernel-batching-with-CUDA-Graphs study reports (paper-measured) a 3-kernel / 100-iteration
pipeline dropping ~25% (~1.00 ms → ~0.75 ms) by collapsing 300 launches into 100 replays,
https://arxiv.org/html/2501.09398v1 (NVIDIA CUDA Graphs docs:
https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html); the persistent-kernel
alternative uses cooperative-groups grid sync,
https://developer.nvidia.com/blog/cooperative-groups/. Background on block- vs path-based STA:
https://en.wikipedia.org/wiki/Statistical_static_timing_analysis and the IBM incremental
block-based SSTA paper,
https://people.eecs.berkeley.edu/~alanmi/research/timing/papers/sta_ibm.pdf.

**Honest positioning.** The paper-measured figures above are each on a *different*
workload/baseline/GPU than ours; they bound what a technique *can* buy, and are **not** a
prediction for our kernel. Where the roadmap pre-estimated ~25% launch-overhead removal for a
comparable pipeline, our *measured* effect was larger (~2×) because our per-level compute is tiny
and launch latency dominated more (E4); the pre-measurement estimate is kept, labeled as an
estimate.

---

## 9. Conclusion and future work

We built the GPU STA primitive no vendor library covers and held it to a strict honesty
contract, and the result reproduced the naive-offload-is-a-net-loss thesis on our own kernel:
against a fair 24-core CPU baseline the naive per-launch level sweep only ties the CPU and loses
on small graphs (E2, E6). The win comes from an engineering change — CUDA-graph capture/replay
collapsing the launch-bound per-level sweep — which buys 1.8–2.1× over per-launch and an honest
1.7–2.6× over 24 cores (E4, E6), bit-exact everywhere, correct on all 11 ISCAS-85 circuits (E7)
and on changing per-corner delays (E8), and accurate to ~1 ppm relative arrival against a
double-precision ground truth (E10). The complementary "buy" half routes BFS/WCC/SSSP to cuGraph,
oracle-checked, while topological sort and DFS (no cuGraph primitive) are built (E9) — the toolbox
over the hammer.

**Future work**, in the project's stated priority order:

- **Iteration 3 — warp-per-node.** A warp-cooperative kernel with degree bucketing and
  `cub::WarpReduce` (the Warp-STAR direction) for the high-fanin intra-warp load-imbalance case,
  gated behind the measured-speedup harness, heeding Warp-STAR's own warning against
  over-engineering the load balancer.
- **A realistic large generator** with skewed fanout, irregular level widths, long-range arcs, and
  reconvergence — and, if obtainable, a large industrial design — to close the generator-realism
  gap (E7) and re-measure the large-scale speedup on a faithful topology.
- **Prior-work comparison** on shared benchmarks (OpenTimer and, if reproducible, a published
  GPU-STA) on the same host, to place this primitive on the same absolute axis as the closest
  prior art.
- **On-device corner generation** to remove the per-corner H2D delay upload from the multi-corner
  sweep (E8), a roofline / achieved-bandwidth analysis, a second-architecture sweep, and committed
  raw GPU CI logs.

---

## References

1. Guo, Huang, Lin. "GPU-Accelerated Static Timing Analysis." ICCAD 2020.
   https://dl.acm.org/doi/10.1145/3400302.3415631 · PDF:
   https://tsung-wei-huang.github.io/papers/iccad20-gpusta.pdf
2. "Warp-STAR: High-performance, Differentiable GPU-Accelerated STA through Warp-oriented Parallel
   Orchestration." https://arxiv.org/html/2603.28381
3. "Accelerating STA using CPU-GPU Heterogeneous Parallelism."
   https://www.researchgate.net/publication/371589918
4. Guo, Huang, Lin, Wong. "GPU-accelerated Path-Based Timing Analysis." DAC 2021.
   https://dl.acm.org/doi/10.1109/DAC18074.2021.9586316 · PDF:
   https://yibolin.com/publications/papers/TIMER_TopK_DAC2021_Guo.pdf
5. "A GPU-Accelerated Framework for Path-Based Timing Analysis." TCAD 2023.
   https://dl.acm.org/doi/10.1109/TCAD.2023.3272274
6. "GCS-Timer: GPU-Accelerated Current Source Model Based STA." DAC 2024.
   https://dl.acm.org/doi/10.1145/3649329.3655983
7. "INSTA: Ultra-Fast, Differentiable, Statistical STA." DAC 2025 (NVIDIA).
   https://research.nvidia.com/labs/electronic-design-automation/papers/yichen_INSTA_dac25.pdf
8. Anzt et al. "Load-balancing Sparse Matrix Vector Product Kernels on GPUs." ACM TOPC 2020.
   https://dl.acm.org/doi/fullHtml/10.1145/3380930
9. LightSpMV: dynamic warp/vector assignment by row length.
   https://www.researchgate.net/publication/276411706
10. Bell & Garland. "Implementing SpMV on Throughput-Oriented Processors." SC09.
    https://www.nvidia.com/docs/io/77944/sc09-spmv-throughput.pdf
11. Kernel-batching-with-CUDA-Graphs study. https://arxiv.org/html/2501.09398v1 · NVIDIA CUDA
    Graphs docs: https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html
12. NVIDIA Cooperative Groups. https://developer.nvidia.com/blog/cooperative-groups/ · CUDA
    Programming Guide (Cooperative Groups):
    https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
13. Statistical STA (block- vs path-based) survey.
    https://en.wikipedia.org/wiki/Statistical_static_timing_analysis · IBM incremental block-based
    SSTA: https://people.eecs.berkeley.edu/~alanmi/research/timing/papers/sta_ibm.pdf

---

*Primary source of every measured number: `docs/lab-notebook.md`, entries E1–E10 (append-only,
each with its exact command and the `maxAbsDiff` oracle check). Design/API details:
`docs/USAGE.md`. Limitations: `docs/red-team-review.md`. Build-vs-buy routing:
`graph_routing/README.md`. Related-work citations: `docs/research/sta-optimization-roadmap.md`.*
