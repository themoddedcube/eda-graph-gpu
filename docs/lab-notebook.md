# eda-graph-gpu — lab notebook

Engineering notebook for the STA primitive. **Discipline** (same as the parent
CUDAadvisor project):

- **Append-only.** Never rewrite history; a correction is a new entry that
  back-references the one it corrects. Dead-ends are kept.
- **Evidence-backed.** Every claim cites a `path:line` or captured command output.
- **Measured ≠ estimated.** A modeled number is labeled *estimated*; a number that
  came out of a real run is labeled *measured*. Never dress one as the other.
- **Reproducible.** A measured result ships with the exact command that produced it
  and a correctness check (here: the `maxAbsDiff` oracle vs the CPU reference).

Hardware for every "measured" entry below, unless stated otherwise:
**NVIDIA H100 PCIe (80 GB), driver 570.124.06, CUDA 12.8** (`nvidia-smi`), CPU
reference single-threaded, `-O2`.

---

## E1 — CPU reference + equivalence oracle (2026-07-18)

**What.** Established the primitive: a struct-of-arrays / CSR timing graph
(`include/timing_graph.h`), a level-ordered CPU reference (`src/sta_cpu.cpp:12`
`staCpu` — forward arrival = max over fanin of `arrival+delay`; backward required =
min over fanout; slack = required − arrival), an order-independent equivalence
comparator (`src/sta_cpu.cpp:56` `maxAbsDiff`), and a deterministic synthetic
layered-DAG generator (`src/sta_cpu.cpp:66` `generateLayeredDag`).

**Evidence / correctness.** Hand-computed 5-node DAG solved on paper matches
`staCpu` exactly (`tests/test_hand_computed.cpp` — period 8, critical path 0→2→4).
`ctest` green.

**Reproduce.** `cmake -S . -B build && cmake --build build && ctest --test-dir build`.

---

## E2 — Measured on H100: the naive GPU port is a *net loss* on small graphs (2026-07-18)

**What.** Ran the level-parallel GPU primitive (`src/sta_gpu.cu`, one kernel launch
per topological level, one thread per node) against the CPU reference on the H100.
First measurement charged the one-time CUDA context init to the single timed call,
which was misleading; `src/main.cpp` now separates **cold** (incl. one-time init)
from **warm** steady-state (min-of-5), which is the honest number when STA is called
repeatedly inside a placement/optimization loop.

**Measured** (min-of-5):

| nodes | arcs | CPU | GPU cold (incl. 1-time init) | GPU warm (per-launch) | oracle |
|---:|---:|---:|---:|---:|:--:|
| 65,536 | 162,565 | 1.27 ms | 289 ms | 1.72 ms · **0.74×** | MATCH (0.0) |
| 262,144 | 652,834 | 5.10 ms | 215 ms | 3.79 ms · **1.35×** | MATCH (0.0) |
| 1,600,000 | 3,991,836 | 32.9 ms | 222 ms | 10.4 ms · **3.18×** | MATCH (0.0) |
| 6,400,000 | 15,980,582 | 132 ms | 309 ms | 33.0 ms · **3.99×** | MATCH (0.0) |

**Finding.** The naive per-launch sweep is **launch-bound**: the ~210–310 ms cold
floor is nearly constant across problem size (it is context init, not compute), and
warm it **loses on small graphs** (0.74× at 65k) and only wins at scale
(crossover ≈ 262k nodes, up to 3.99× at 6.4M). This is the CUDAadvisor thesis — a
naive GPU offload is often a net loss — reproduced on our own primitive.
The oracle is bit-exact (`maxAbsDiff = 0.000e+00`) at every size.

**Reproduce.** `make && ./sta 256 1024 5` (and 128×512, 400×4000, 800×8000), or
`python3 bench/bench.py`. Correctness = the `MATCH` line (exit nonzero on mismatch).

---

## E3 — Improvement iteration 1: on-device period reduction + fused slack (2026-07-18)

**What.** Removed the mid-pipeline host round-trip. Previously we synced, copied the
whole `arrival` array to the host, and reduced the period in a host loop between the
forward and backward sweeps. Now `poMaskKernel` (`src/sta_gpu.cu:53`) masks non-POs
to −inf and `cub::DeviceReduce::Max` produces the period on-device, and slack is
**fused** into `requiredKernel` (`src/sta_gpu.cu:63`). One `cudaDeviceSynchronize`
+ one D2H of results now covers the whole pipeline.

**Correctness (measured).** The math is identical (max/min/subtract), so equivalence
holds by construction — and is confirmed on-device: oracle `MATCH (0.0)` at every
size in E2's table (those runs already include this change). Chosen because it is the
lowest-risk item and the enabler for E4.

**Reproduce.** Same as E2; the oracle line is the correctness check.

---

## E4 — Improvement iteration 2: CUDA-graph capture/replay — measured ~2× (2026-07-18)

**What.** Real STA re-evaluates the same topology many times (incremental timing).
Added a persistent plan (`src/sta_gpu.cu` `StaGpuPlan` / `staGpuPlanCreate` /
`staGpuPlanRun` / `staGpuPlanDestroy`) that stream-captures the entire
forward + period-reduce + backward chain into a **CUDA graph once**, then replays it
with a single `cudaGraphLaunch` — collapsing the ~2·`numLevels` (≈800–1600) per-level
kernel launches that E2 showed were the bottleneck.

**Measured** (min-of-5, same H100):

| nodes | CPU | GPU warm (per-launch) | **GPU graph-replay** | replay vs per-launch | oracle |
|---:|---:|---:|---:|---:|:--:|
| 65,536 | 1.27 ms | 1.72 ms · 0.74× | **0.82 ms · 1.54×** | **2.09×** | MATCH (0.0) |
| 262,144 | 5.10 ms | 3.79 ms · 1.35× | **1.81 ms · 2.82×** | **2.09×** | MATCH (0.0) |
| 1,600,000 | 32.9 ms | 10.4 ms · 3.18× | **5.74 ms · 5.73×** | **1.80×** | MATCH (0.0) |
| 6,400,000 | 132 ms | 33.0 ms · 3.99× | **16.1 ms · 8.18×** | **2.05×** | MATCH (0.0) |

**Finding.** CUDA-graph replay buys **1.8–2.1× over the per-launch path**, pushes the
CPU-crossover **below 65k** (0.74× → 1.54×), and scales to **8.18× vs CPU** at 6.4M
nodes — bit-exact everywhere. The graph *capture* is a one-time cost (1.3–16.7 ms,
`main.cpp` reports it separately), amortized over repeated evaluations. This directly
confirms E2's diagnosis that the primitive was launch-bound.

**Estimated vs measured note.** `docs/research/sta-optimization-roadmap.md` had
*estimated* ~25% launch-overhead removal for a comparable pipeline (paper-derived).
The *measured* effect on our kernel is larger (~2×) because our per-level compute is
tiny, so launch latency dominated more than the cited pipeline. The roadmap's
pre-measurement estimates are kept, labeled as estimates.

**Reproduce.** `make && ./sta 400 4000 5` → the `graph-replay` line; or
`python3 bench/bench.py`. Correctness = the `MATCH` oracle line.

---

## E5 — Edge/base-case + property test suite (2026-07-18)

**What.** Added `tests/test_edge_cases.cpp`: hand-built degenerate graphs with
arithmetic worked out in comments — single node (PI≡PO), a single all-PI/PO level, a
pure chain, **zero-delay arcs** (exercises the PO-masked period reduction, which is
correct even when an internal node ties a PO), and a **high-fanin star** (fanin
reduction at degree > 1) — plus structural property checks on generated graphs across
seeds/sizes (PI arrival 0; PO required = period; slack = required − arrival; arrival =
independently-recomputed max over fanin; period = max PO arrival), generator
determinism, and a **GPU cross-check of both `staGpu` and the CUDA-graph plan** on
every case when a device is present.

**Evidence.** `ctest --test-dir build` → 5/5 pass on the H100; `test_edge_cases`
reports "GPU cross-checked on N graphs". On a CPU-only box the GPU branches skip
honestly (still exit 0), matching CI's default job.

**Reproduce.** `ctest --test-dir build --output-on-failure`.

---

## E6 — Fair 24-core CPU baseline retires the "8× vs 1-core" headline (2026-07-18)

**Corrects E2 and E4.** Those entries reported the GPU win **vs a single CPU thread**
(up to 8.2×). A hostile-reviewer pass (`docs/red-team-review.md`, criticism #1)
correctly calls that a strawman: real STA (OpenTimer et al.) uses every core. Added
`src/sta_cpu_mt.cpp` `staCpuParallel` — the SAME level decomposition via OpenMP,
**bit-identical to `staCpu`** (asserted: `tests/test_oracle_consistency.cpp`
`maxAbsDiff==0`) — and made it the baseline in `main.cpp` / `bench.py`.

**Measured** (min-of-15, H100 vs all 24 cores):

| nodes | CPU 1-core | CPU 24-core | GPU per-launch | GPU graph-replay | replay vs 24-core |
|---:|---:|---:|---:|---:|---:|
| 262,144 | 5.08 ms | 3.53 ms | 3.95 ms | 1.82 ms | 1.9× |
| 1,600,000 | 32.8 ms | 16.2 ms | 12.4 ms | 6.19 ms | 2.6× |
| 6,400,000 | 131 ms | 27.6 ms | 32.5 ms | 15.9 ms | 1.7× |

**Finding.** Against a fair all-core baseline the honest win is **~1.7–2.6×**, not 8×,
and the **naive per-launch GPU merely ties the 24-core CPU** (0.85–1.31×) — the
CUDA-graph optimization (E4) is what actually earns the GPU its keep. The headline
number changed by ~4× once the baseline was made fair; keeping E2/E4 above, uncorrected
in place, per append-only discipline.

**Two methodology traps caught while doing this (both fixed):**
- *Silent single-threading.* The OpenMP thread count followed an ambient
  `OMP_NUM_THREADS` / inherited CPU-affinity mask, so a subprocess baseline silently
  ran on 1 thread while claiming "24-core". Fixed by setting the count explicitly to
  `omp_get_num_procs()` (`src/sta_cpu_mt.cpp`, `src/main.cpp`) — deterministic now.
- *Parser matched the wrong line.* `bench.py`'s `(\d+)-core` regex matched the
  *1-core* line first and mislabeled it the multi-core baseline. Fixed to select the
  max-core line. (A reminder that "measured" includes measuring the harness.)

**Estimated vs measured caveat.** The 24-core baseline is memory-bandwidth-bound and
**noisy on this shared cloud box**: min-of-15 `replay vs 24-core` moved ±0.4× between
runs with CPU contention (a stray 96%-CPU neighbor process was visible during one
sweep). The numbers above are approximate; the GPU times are stable. A quiet, named,
isolated host is the next step (tracked in the red-team review, items #9/#11).

**Reproduce.** `python3 bench/bench.py --reps 15` → the `cpuN`/`replay_vs_N` columns.

---

## E7 — STA validated on REAL ISCAS-85 netlists + honest topology profile (2026-07-18)

**Addresses red-team #2** ("every number is on a synthetic GPU-friendly DAG"). Added
real-netlist ingestion — a general DAG **levelizer** (`src/circuit.cpp` `levelize`:
longest-path-from-source levels, cycle rejection) and an **ISCAS-85 `.bench` reader**
(`readBench`) — plus a topology profiler (`profileGraph`, `./build/profile`). Fetched
all 11 public-domain ISCAS-85 circuits into `bench/circuits/` (provenance in its README).

**Correctness (measured).** STA is **bit-exact (oracle MATCH) on all 11 real circuits**,
CPU and GPU (`tests/test_bench_circuits.cpp`, 6/6 ctest green). c17 is checked against a
hand-derived result (11 nodes, 12 arcs, 4 levels, 5 PI, 2 PO, **period 6**).

**Topology profile — real vs synthetic (`./build/profile`):**

| graph | nodes | levels | meanW | maxW | meanFanin | maxFanin | maxFanout |
|---|---:|---:|---:|---:|---:|---:|---:|
| c432 | 196 | 18 | 10.9 | 36 | 1.71 | 9 | 9 |
| c2670 | 1426 | 33 | 43.2 | 233 | 1.46 | 5 | 233 |
| c7552 | 3719 | 44 | 84.5 | 355 | 1.65 | 5 | 207 |
| syn-mid | 65,536 | 128 | 512 | 512 | 2.48 | 4 | 11 |
| syn-big | 1,600,000 | 400 | 4000 | 4000 | 2.49 | 4 | 14 |

**Findings (honest, both directions):**
1. Correctness generalizes to real, irregular topology — not a synthetic artifact.
2. Real *available* benchmarks are tiny (≤3.7k gates) with narrow irregular levels, so
   the GPU is launch-bound and **the CPU is the right tool there** — the GPU regime is
   the *large-scale* graph (10⁵–10⁷ nodes) that modern industrial designs reach.
3. **The synthetic generator is not fully faithful:** its levels are perfectly uniform
   (meanW == maxW) and fanin is banded ≤4, whereas real circuits have irregular widths
   and **high-fanout hubs** (maxFanout up to ~300) + higher maxFanin (up to 16). So the
   large-scale synthetic numbers (E4/E6) live on a *more regular* graph than reality —
   a real limitation. **Next:** a realistic large generator (skewed fanout, irregular
   widths, reconvergence) and, if obtainable, a large industrial design.

**Reproduce.** `./build/profile bench/circuits`; `ctest -R test_bench_circuits`.

---

## E8 — Changing-input replay: multi-corner re-evaluation (red-team #3 + #4) (2026-07-18)

**Corrects E4's framing** and closes red-team **#3** ("incremental STA contradicts a
static CUDA graph") and **#4** ("replay recomputes a bit-identical answer"). E4 called
replay "incremental timing", which the reviewer rightly flagged: incremental timing
changes topology, but a CUDA graph is topology-static, and re-running the same delays N
times models nothing. The honest workload is **multi-corner / Monte-Carlo STA**: ONE
fixed topology re-evaluated under many different ARC DELAYS.

**What.** Added `staGpuPlanUpdateDelays(plan, finDelay, foutDelay)` (`src/sta_gpu.cu`):
it `cudaMemcpy`s a new delay set into the plan's device buffers — the captured graph
references the same pointers, so **no re-capture** is needed, only the next replay uses
the new delays. Added `withFinDelay` (`src/circuit.cpp`) to build the matching CPU
oracle for a corner. `main.cpp` runs a K-corner sweep; `staGpuPlanRun` is now genuine
work per corner.

**Correctness (measured, `tests/test_corner_replay.cpp`, 7/7 ctest green).** For each of
6 non-uniform corners (every arc scaled by a random factor in [0.5,1.5]): the GPU result
(a) **matches `staCpu` on that corner's delays** (`maxAbsDiff ≤ 1e-2`) and (b) **differs
from the base corner** (`maxAbsDiff > 0.5`) — so it is provably not re-running one
answer. A wrong-size update is rejected and does not corrupt the plan.

**Measured corner sweep** (H100, 1.6M-node graph, K=16 delay sets):

| | time | note |
|---|---:|---|
| CPU 24-core (re-solve each corner) | 208.5 ms | fair all-core baseline |
| GPU plan (updateDelays + replay per corner) | 171.4 ms | **1.22×**, results vary (`|c0−clast|=153`) |

**Honest read.** Only ~1.2× here, *not* the 1.7× of a single evaluation, because the
sweep **includes the per-corner H2D upload of the delay arrays** (~2·numArcs floats ×
K) — a real cost we deliberately time, and exactly the transfer the reviewer's H2D point
cares about. The win is modest but genuine, and correctness on changing input is exact.
A future step generates corners on-device to remove that upload.

**Reproduce.** `./build/sta 400 4000` → the `corner sweep` line; `ctest -R test_corner_replay`.

---

## E9 — Basic graph traversals → cuGraph, oracle-checked (the "buy" half) (2026-07-19)

**What.** Built the complement to the STA primitive: the standard EDA graph traversals
routed to **cuGraph** where it has a real parallel primitive, kept on our own reference
where it doesn't (`graph_routing/`). Installed RAPIDS cuGraph 26.06 into `.venv-rapids`
(gitignored); it imports in ~1.3 s and runs on the H100.

**The mapping (honest build-vs-buy).** cuGraph provides BFS, WCC/SCC, SSSP, PageRank —
but **no topological sort and no DFS** (both inherently sequential). So:
- BFS / WCC / SSSP / PageRank / SCC → **buy** (route to cuGraph).
- Topological sort (= the STA levelization) and DFS → **build** (our levelizer /
  CPU reference). No cuGraph primitive exists to route to.

**Validation (measured, `graph_routing/test_routing.py`).** Each routed primitive runs
on the GPU and is checked **bit-for-bit against the pure-Python oracle** (`graphkit.py`)
on a synthetic graph + **all 11 ISCAS-85 circuits**. Result: **every case MATCHes**
("ROUTING OK — every cuGraph result matched the oracle"). Sample:

```
c2670: bfs MATCH (reach=5)   wcc MATCH (80 comps)  sssp MATCH
c7552: bfs MATCH (reach=8)   wcc MATCH (5 comps)   sssp MATCH
```

**Two real-netlist convention findings (both fixed, not cuGraph defects):**
1. **WCC on isolated nodes.** Real circuits have unconnected primary inputs; the
   union-find oracle counts each as its own component, but cuGraph never sees a node
   absent from the edgelist. c2670 exposed it (oracle 80 vs cuGraph-lumped 5). Fix:
   the adapter gives each absent node its own singleton component — then 80 == 80.
2. **SSSP on parallel arcs.** A gate with a repeated input yields parallel arcs; the
   arbitrary per-index weights made the min-path ambiguous (c3540 mismatch). Fix: run
   SSSP on the deduped simple graph, so both sides see identical weights.

**Reproduce.** `.venv-rapids/bin/python graph_routing/test_routing.py`.
Without cuGraph the reference oracle still runs and routing reports "wired, GPU
validation skipped" — never a fake pass.

---

## Open hardening backlog (from the red-team review, 2026-07-18)

`docs/red-team-review.md` lists 12 criticisms. Status:
- **#1 fair baseline — DONE** (E6).
- **#2 real netlists — PARTIALLY DONE** (E7): ingestion + bit-exact correctness on 11
  ISCAS-85 circuits + real-vs-synthetic topology profile shipped. *Remaining:* a
  realistic large generator (skewed fanout / irregular widths) and a large design.
- **cycle rejection** (part of the "tests aren't adversarial" item) — DONE (E7 test).
- **#3/#4 changing-input replay — DONE** (E8): `staGpuPlanUpdateDelays` + multi-corner
  re-evaluation; results provably change per corner and match the CPU oracle; the
  per-corner H2D delay upload is included in the measured sweep (partial answer to the
  H2D item too).

Next, in priority order: **#5** a double-precision ground truth + bounded fp32 error;
then a realistic large generator, prior-work comparison, a roofline, on-device corner
generation (to remove the delay-upload cost), and committed GPU CI logs.
