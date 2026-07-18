# Red-team review — `eda-graph-gpu` (adversarial pre-submission critique)

**Reviewer stance:** hostile expert peer reviewer (EDA/CAD + GPU systems + benchmarking
methodology). The goal is to enumerate every attack a skeptic will mount on the core
claim so the authors can neutralize it *before* submission.

**Core claim under attack:** *a level-parallel max-plus STA GPU primitive that is
bit-exact vs a CPU reference and, with CUDA-graph replay, runs up to ~8.2× vs CPU on an
H100.*

**Scope note / working-tree state.** As of this review the repo has uncommitted changes
(`git status`: `M src/main.cpp`, `M include/sta.h`, `M CMakeLists.txt`, `M Makefile`,
`?? src/sta_cpu_mt.cpp`) that add an OpenMP multi-threaded CPU baseline
(`staCpuParallel`). **That baseline is never measured**: `bench/results.csv`,
the `README.md` table (`README.md:58-63`), and `docs/lab-notebook.md` (E2/E4 tables)
still report only the single-thread CPU and still literally say *"Speedups are vs a
single-thread CPU reference"* (`README.md:56`, `docs/lab-notebook.md:16`). Several
criticisms below are therefore "code exists, evidence does not."

---

## Summary table (most-damaging first)

| # | Criticism | Severity | Status | Fix artifact |
|---|-----------|:--------:|:------:|--------------|
| 1 | Headline speedup is vs a **single-thread** CPU; a real STA engine uses all cores | Critical | PARTIAL (MT code added but **unmeasured**, headline still 1-thread) | Report GPU vs OpenTimer/OpenMP on the same host; retire the 1-core headline |
| 2 | Benchmark is a **synthetic, uniform-width random DAG** — no real netlist | Critical | NONE | ISCAS-85/89, EPFL, ITC, TAU-contest / superblue netlists; report level-width & fanout profiles |
| 3 | "Incremental STA" framing **contradicts** the CUDA-graph replay it's sold on | Critical | NONE | An actual incremental path (edit → re-time cone); or drop the incremental claim |
| 4 | Replay re-computes a **bit-identical** result N times — no input ever changes | High | NONE | Feed changing PI/delay inputs per replay; show correctness + speed under real updates |
| 5 | fp32 "bit-exact" is **self-referential**; no double-precision ground truth | High | NONE | Double-precision reference; report abs/rel error vs it on deep paths |
| 6 | Not comparable to **prior GPU-STA** (Warp-STAR, ICCAD'20); apples-to-oranges | High | PARTIAL (cited, not run) | Same-netlist head-to-head vs OpenTimer + a published GPU-STA |
| 7 | **What's timed**: H2D excluded from every headline number; min-of-5 hides tail; CPU unnamed | High | PARTIAL | Full end-to-end timing incl. H2D; report mean/median/p99; name the CPU |
| 8 | Win is a **memory-bandwidth / launch-overhead** artifact, not an algorithmic result | High | PARTIAL | Roofline / bandwidth-efficiency vs peak; per-node work sweep |
| 9 | **Correctness coverage** gaps: no cycle rejection, oracle ignores slack, tiny max-fanin | Medium | PARTIAL | Adversarial suite; put slack/period in the oracle; huge-fanin & malformed-input tests |
| 10 | **Generality**: one GPU, one arch (sm_90); no width/depth sensitivity study | Medium | NONE | Second GPU (A100/consumer); sweep depth×width independently |
| 11 | **Reproducibility**: GPU numbers never run in CI; no committed run logs | Medium | PARTIAL | Commit raw stdout + `nvidia-smi`/`nvcc` capture; nightly self-hosted GPU CI |
| 12 | Generator is **artificially local** (arcs only from L-1/L-2), suppressing reconvergence | Medium | NONE | Long-range arcs + realistic reconvergence, or use real graphs (#2) |

---

## 1. The baseline is a single CPU thread — the whole speedup is against a strawman
**(Critical)**

- **A reviewer will say:** "Your 8.2× is against *one* CPU core while a production STA
  engine (OpenTimer, Taskflow) saturates all cores; normalize to a full-socket baseline
  and most of the win evaporates — you may even be *slower* per watt/socket."
- **Why it bites:** it undermines the headline number directly. Your own roadmap cites
  ICCAD'20 ("one GPU faster than OpenTimer of **40 CPUs**") and Warp-STAR ("162× over
  OpenTimer") — those are the yardsticks. An 8.2× over 1 core is roughly break-even
  against ~8–16 cores of the *same* embarrassingly-parallel level sweep, i.e. no result.
- **Status — PARTIAL.** An OpenMP baseline was just added (`src/sta_cpu_mt.cpp`
  `staCpuParallel`, wired into `src/main.cpp:42-55,66-68,86-88,107-111`) but **it has
  never been run into any reported table**: `bench/results.csv` has no MT column, and
  `README.md:56` + `docs/lab-notebook.md:16` still say *single-thread*. Worse, the MT
  baseline itself is weak: it opens/closes an OpenMP parallel region *per level*
  (`src/sta_cpu_mt.cpp:35,58` — ~800 fork/joins for the default run), so its own
  overhead will understate what a task-parallel engine achieves, letting a skeptic
  attack it from the *other* side ("your multicore baseline is deliberately bad").
- **Fix artifact:** (a) measure GPU vs `staCpuParallel` on a *named* CPU at all sizes and
  put the `Nx-core` column in the README/CSV/notebook; (b) additionally benchmark against
  **OpenTimer** on the *same host and same netlists* so the baseline is a real engine, not
  your own loop; (c) restructure the OpenMP baseline to a single parallel region with a
  per-level barrier (or Taskflow) so it is a *strong* multicore baseline; (d) retire the
  "vs single-thread" headline entirely — report the multicore speedup as the primary
  number and 1-core only as a secondary datapoint.

## 2. The benchmark is a synthetic, uniform-width random DAG — not a real circuit
**(Critical)**

- **A reviewer will say:** "Every number is on `generateLayeredDag`, which produces
  *perfectly uniform* levels (exactly `widthPerLevel` nodes each) with iid fanin ≤ 4 from
  the immediately preceding one or two levels — the single most GPU-favorable topology
  possible. Real netlists have wildly non-uniform level widths, heavy-tailed fanout,
  deep reconvergence, and long-range arcs. Show me ISCAS-85/89, EPFL, ITC, or a
  TAU-contest design or I don't believe the win transfers."
- **Why it bites:** it attacks external validity of *every* measurement. GPU throughput
  on a level sweep is dominated by the width of each level; a generator that guarantees
  every level is 4,000–8,000 nodes wide manufactures full occupancy on every launch.
  Real designs have many thin levels (tens of nodes) where the GPU is launch-bound and
  *loses* — exactly the regime your own 65k-node row already shows the naive path losing
  (`README.md:60`). The 8.2× is reported at the *widest* configuration (8,000/level),
  the least realistic point.
- **Evidence of the bias:** `src/sta_cpu.cpp:66-103` — `numNodes = numLevels*widthPerLevel`,
  every `levelStart[L] = L*widthPerLevel` (`:75`), fanin `1 + rng()%maxFaninPerNode`
  with `maxFaninPerNode=4` fixed (`src/main.cpp:29`), sources only from `[max(0,L-2), L)`
  (`:89,93`). No fanout skew, no wide-then-narrow profile, no primary-output cone
  structure.
- **Status — NONE.** `docs/USAGE.md:142-145` and `:378-380` *admit* real ingestion is
  Stage 2 / not implemented, but honesty about the gap is not a substitute for the
  experiment; the paper's claim needs a real netlist.
- **Fix artifact:** ingest real timing graphs (ISCAS-85/89, EPFL "arithmetic"/"random_control",
  ITC'99, and at least one large TAU-contest / superblue design), rerun the whole table,
  **and** publish each design's level-width histogram and fanin/fanout degree
  distribution next to the speedup so a reader can see the level profile the GPU actually
  faced. Report the *worst* design, not the best.

## 3. The "incremental STA" story contradicts the CUDA-graph technique it rests on
**(Critical)**

- **A reviewer will say:** "You justify CUDA-graph replay by 'real STA re-evaluates the
  same topology many times (incremental timing inside a placement loop)'
  (`include/sta.h:29-33`, `docs/USAGE.md:202-206`), but incremental timing is triggered
  by *netlist edits that change the topology* — and a CUDA graph is topology-static, so
  the very use case you invoke is the one that forces a re-capture/re-instantiate. You're
  selling the technique on the workload that breaks it."
- **Why it bites:** it dismantles the framing of the flagship 1.8–2.1× / 8.2× result. If
  the topology is truly static, it isn't "incremental STA" (that's just repeated *full*
  re-evaluation, a rare need); if it's genuinely incremental, `staGpuPlanCreate` must be
  re-run on every structural edit and the amortization argument collapses.
- **Status — NONE.** There is no incremental API; `staGpuPlanRun` replays a fixed graph
  (`src/sta_gpu.cu:262-278`). The roadmap lists incremental STA as unstarted P8
  (`docs/research/sta-optimization-roadmap.md:194`).
- **Fix artifact:** implement and measure a real incremental step — apply a delay/edge
  edit, re-time only the affected fan-in/fan-out cone, and compare GPU cone-retiming vs
  CPU cone-retiming — *or* drop the "incremental" justification and honestly reframe
  replay as "repeated full re-evaluation of an unchanged graph (e.g., Monte-Carlo /
  corner sweeps)," which is a defensible but much narrower claim.

## 4. Replay measures re-computing a bit-identical answer with no changing input
**(High)**

- **A reviewer will say:** "`staGpuPlanRun` reads the same uploaded delays every time and
  produces the same output — your test even asserts `maxAbsDiff(a,b)==0` for two replays
  (`tests/test_edge_cases.cpp:99`). You're benchmarking the cost of recomputing a
  constant. Where do updated arrival/delay inputs enter between replays?"
- **Why it bites:** it makes the replay benchmark a micro-benchmark of kernel-launch
  plumbing, not of a workload. There is no host- or device-side path to mutate PI arrival
  times, arc delays, or the graph between `staGpuPlanRun` calls, so N replays model
  nothing a real tool does N times.
- **Status — NONE.** `StaGpuPlan` holds device buffers filled once at create
  (`src/sta_gpu.cu:205-219`); `staGpuPlanRun` only launches + copies out (`:262-278`).
- **Fix artifact:** add an input-update entry point (e.g., overwrite `dFinDelay` / PI
  seeds before each replay — CUDA graphs allow node-parameter updates via
  `cudaGraphExecKernelNodeSetParams`, or just re-`memcpy` the delay array), then show the
  replay path still produces the correct (changing) result and still beats CPU when the
  inputs actually vary.

## 5. fp32 "bit-exact" is self-referential — there is no ground truth
**(High)**

- **A reviewer will say:** "`maxAbsDiff == 0` proves the GPU reproduces the *fp32* CPU
  reference, both of which accumulate the same rounding error. It says nothing about
  correctness. Over 800 levels arrival reaches ~2,924 (`bench/results.csv:5`,
  `period=2923.822`); in fp32 that carries ~2e-4 relative error (~0.5 ULP·paths). Show me
  the error vs a double-precision solution."
- **Why it bites:** "bit-exact" is the paper's correctness pillar; a skeptic reframes it
  as "two identically-wrong computations agree." Slack (`required − arrival`) and
  `required − delay` are subtractions where catastrophic cancellation *can* bite when
  slack ≈ 0 on the critical path — precisely the values STA cares about.
- **Status — NONE.** No `double` reference anywhere; oracle compares fp32 to fp32
  (`src/sta_cpu.cpp:56-64`). The README calls this "bit-exact" (`README.md:53`).
- **Fix artifact:** add a `staCpuDouble` reference and report `max_abs` and
  `max_rel` error of the fp32 GPU result vs it, per array, at the deepest configuration;
  characterize worst-case slack error near the critical path. If error is bounded and
  tiny, that is a *stronger* correctness claim than "matches our own fp32."

## 6. The numbers are not comparable to prior GPU-STA work
**(High)**

- **A reviewer will say:** "Warp-STAR reports superblue1 at 25 ms and 2.4× over the prior
  GPU-Timer / 162× over OpenTimer; ICCAD'20 reports 3.69× over 40-core OpenTimer. Your
  8.2× is over one CPU core on a synthetic graph — I cannot place you on the same axis,
  so I cannot tell if this is state-of-the-art or a toy."
- **Why it bites:** a systems/EDA venue expects positioning against the closest published
  baselines on shared benchmarks. Without a common design and a common CPU engine, the
  contribution is unmeasurable relative to the field.
- **Status — PARTIAL.** The prior art is carefully cited
  (`docs/research/sta-optimization-roadmap.md:66-98`) and honestly flagged as
  different-workload/GPU/baseline (`:17-19`), but none of it is *run*.
- **Fix artifact:** at minimum, run OpenTimer and (if obtainable) a published GPU-STA
  implementation on the same netlists/host and report your primitive on the same axis;
  if a GPU baseline can't be reproduced, at least match the *benchmark designs* used by
  Warp-STAR/ICCAD'20 so absolute ms are comparable.

## 7. What's actually timed hides the real cost (transfer, tail, unnamed CPU)
**(High)**

- **A reviewer will say:** "The graph-replay headline excludes the H2D upload of the
  whole graph (~250 MB of CSR/delays at 6.4M nodes), counting it only as a one-time
  'capture' cost; you report min-of-5, which reports the best run and hides tail
  variance; and you never state the CPU model, so the speedup denominator is unknown."
- **Why it bites:** three separate honesty attacks on the same number. (a) The CPU path
  never transfers anything; excluding H2D from the GPU number is only fair if the graph
  truly never re-uploads — see #3/#4. (b) A placement inner loop cares about *throughput*
  (mean) and *worst case* (p99), not the min. (c) A speedup with an unnamed baseline CPU
  is uninterpretable.
- **Status — PARTIAL.** Cold-vs-warm separation and the CPU-fallback honesty are genuinely
  good (`src/main.cpp:57-62,85`); capture time *is* reported separately (`:107-111`). But
  D2H of all three result arrays *is* inside every replay (`src/sta_gpu.cu:271-276`) while
  H2D is not; min-of-5 is the only statistic (`src/main.cpp:39,53,80,103`); and the
  lab-notebook names the GPU/driver/CUDA (`docs/lab-notebook.md:14-16`) but **not the
  CPU**.
- **Fix artifact:** add an end-to-end column that includes H2D+compute+D2H for a
  cold/one-shot evaluation (the honest number when topology changes); report
  mean/median/p99 alongside min; and record the exact CPU (model, sockets, cores, clock,
  RAM, compiler flags — note the Makefile uses `-O2`, not `-O3 -march=native`,
  `Makefile:8`) in the notebook.

## 8. The "win" is a bandwidth / launch-overhead artifact, not an algorithm result
**(High)**

- **A reviewer will say:** "Per-node work is a handful of adds over ≤4 fanin
  (`src/sta_gpu.cu:41`); this is a memory-bound max-plus SpMV. An H100 has ~3.35 TB/s vs
  a CPU's ~50–100 GB/s — a ~30–60× bandwidth edge. Getting only 8.2× means you're leaving
  most of the device on the table, and the 'speedup' is just running a memory-bound
  reduction on higher-bandwidth silicon — no EDA-specific insight."
- **Why it bites:** it reframes the contribution as "we moved a bandwidth-bound kernel to
  a bigger-bandwidth chip," which is neither novel nor efficient. It also implies your own
  kernel is under-optimized (uncoalesced `arrival[finFrom[p]]` gather,
  `docs/research/sta-optimization-roadmap.md:31`), so a stronger CPU or a better GPU
  kernel would both move the number a lot.
- **Status — PARTIAL.** The roadmap diagnoses the launch-bound / uncoalesced nature
  honestly (`:29-41`) and the CUDA-graph work targets launch overhead, but there is no
  roofline / achieved-bandwidth analysis anywhere.
- **Fix artifact:** report achieved GB/s vs device peak (roofline) for the forward/backward
  kernels, and sweep per-node work (vary `maxFaninPerNode` well beyond 4, e.g. 16/64/256)
  to show whether the win is stable or purely a thin-kernel/bandwidth effect. If it's
  bandwidth-bound, say so and frame the contribution accordingly.

## 9. Correctness coverage is not adversarial enough
**(Medium)**

- **A reviewer will say:** "Your tests validate the happy path. There's no cycle-rejection
  test (the primitive silently trusts acyclicity — `docs/USAGE.md:62,68`), no huge-fanin
  case (max tested is 8, `tests/test_edge_cases.cpp:198`), no malformed-input handling,
  and your oracle `maxAbsDiff` compares only `arrival` and `required` — **slack and period
  are not in the equivalence check** (`src/sta_cpu.cpp:59-62`), so a GPU slack bug would
  pass."
- **Why it bites:** it attacks the reliability claim. "Bit-exact" is the headline, yet the
  comparator that enforces it ignores two of the four output arrays; the property test
  checks `slack == required − arrival` (`tests/test_edge_cases.cpp:117`) but never
  GPU-slack vs CPU-slack directly. No adversarial degree/depth pathologies, no cycle or
  disconnected-graph rejection, no fp precision stress.
- **Status — PARTIAL.** There is a decent edge/property suite (`tests/test_edge_cases.cpp`)
  and a GPU cross-check, but it is not adversarial and the oracle is incomplete.
- **Fix artifact:** (a) extend `maxAbsDiff` (or add a companion) to cover `slack` and
  `period`; (b) add tests for a node with thousands of fanin, extreme depth-1 vs
  width-1 shapes, duplicate/parallel arcs, and — since the API trusts acyclicity — an
  explicit documented contract test or a debug-mode validator that *rejects* a cycle or
  a same-level arc instead of silently producing garbage.

## 10. One GPU, one architecture, no sensitivity study
**(Medium)**

- **A reviewer will say:** "Every number is a single H100 PCIe at sm_90. Does the win hold
  on an A100, an L40S, or a consumer card with far less bandwidth? Is the crossover point
  a property of the algorithm or of this one chip?"
- **Why it bites:** generality of the claim. The launch-bound/bandwidth-bound behavior is
  device-specific; the crossover node count (currently ~65k, `README.md:60`) will move
  with launch latency and bandwidth, so a single device can't support a general claim.
- **Status — NONE.** Only H100 PCIe measured (`docs/lab-notebook.md:14-16`).
- **Fix artifact:** rerun the sweep on at least one more architecture (ideally an A100 and
  a lower-bandwidth card) and report how the CPU-crossover and peak speedup shift; separate
  the depth and width axes so the reader sees which regime wins where.

## 11. The GPU numbers are never reproduced in CI and ship without raw evidence
**(Medium)**

- **A reviewer will say:** "The headline table exists only as author-typed markdown and a
  4-row CSV; the default CI never compiles or runs the GPU path (`.github/workflows/ci.yml:30-40`),
  the GPU job is opt-in and gated (`:46-47`), and no raw run log, `nvidia-smi`, or `nvcc
  --version` capture is committed. I cannot verify a single GPU number."
- **Why it bites:** artifact-evaluation / reproducibility is table stakes at top venues.
  "Measured, not modeled" is asserted but the measurement leaves no committed trace beyond
  a hand-written table.
- **Status — PARTIAL.** The CPU CI is honest and the fallback truly never fabricates a GPU
  number (good), but the GPU evidence chain is author-trust only.
- **Fix artifact:** commit the raw `./sta` stdout for every reported size plus captured
  `nvidia-smi`, `nvcc --version`, and CPU `lscpu` output under `docs/`; wire a nightly
  self-hosted GPU CI that regenerates `bench/results.csv` and diffs it against the
  committed table so drift is caught.

## 12. The generator is artificially local, suppressing reconvergence and long arcs
**(Medium)**

- **A reviewer will say:** "Fanin is drawn only from the previous one or two levels
  (`src/sta_cpu.cpp:89`), so there are no long-range arcs and little reconvergence — real
  timing graphs have a PI or early pin driving pins many levels deep and heavy
  reconvergent fanout. Your DAG is a nearly-banded matrix, which flatters both the level
  sweep and the CPU cache behavior."
- **Why it bites:** it compounds #2 — even setting aside uniform width, the *connectivity*
  is unrealistically banded, so memory-access irregularity (the thing that hurts GPUs on
  real netlists) is understated.
- **Status — NONE.** Banding is hard-coded (`loLevel = max(0, L-2)`, `src/sta_cpu.cpp:89`).
- **Fix artifact:** either replace with real netlists (#2), or extend the generator with
  tunable long-range-arc probability and reconvergence, and report results across that
  knob — showing the win is not an artifact of the banded structure.

---

## Bottom line for the authors

The engineering hygiene (honest cold/warm split, CPU-fallback that never fabricates a
number, cited prior art, append-only notebook) is genuinely above average and will earn
goodwill. But the *scientific* claim rests on four load-bearing weaknesses a hostile
reviewer will go straight for: **(1)** a single-thread baseline, **(2)** a synthetic
GPU-favorable graph with no real netlist, **(3/4)** an "incremental STA" justification
that both contradicts the static CUDA graph and re-computes an unchanging answer, and
**(5)** an fp32 "bit-exact" claim with no ground truth. None of the current artifacts
close these. Neutralize 1–5 (fair multicore + OpenTimer baseline, real netlists with
published degree/level profiles, a true incremental or honestly-reframed replay, and a
double-precision error bound) before submission; 6–12 are the follow-on hardening.

---

## Author status log (progress against this review)

- **2026-07-18 — #1 (single-thread baseline): FIXED.** Added `staCpuParallel` (OpenMP,
  bit-identical to the reference), made the all-core CPU the baseline in `main.cpp` /
  `bench.py`. Honest headline is now ~1.7–2.6× vs 24 cores (was 8× vs 1 core); naive
  per-launch merely ties the CPU. See lab-notebook E6.
- **2026-07-18 — #2 (synthetic-only benchmark): PARTIALLY FIXED.** Added a DAG
  levelizer + ISCAS-85 `.bench` reader (`src/circuit.cpp`) and a topology profiler;
  STA is bit-exact on all 11 real ISCAS-85 circuits, and the real-vs-synthetic level/
  degree profile is published. Remaining: a realistic *large* generator (skewed
  fanout, irregular widths) + a large industrial design. See lab-notebook E7.
- **2026-07-18 — adversarial tests / cycle rejection: FIXED.** `levelize` rejects
  cycles; `test_bench_circuits` asserts it.
- Open next: #3/#4 changing-input replay, #5 double-precision ground truth, then
  prior-work comparison, H2D accounting, roofline, committed GPU CI logs.
- **2026-07-18 — #3 (incremental framing) + #4 (identical-answer replay): FIXED.**
  Added `staGpuPlanUpdateDelays` so a captured plan is re-evaluated under many timing
  CORNERS (new arc delays, same topology — a valid static CUDA graph). Reframed the
  workload honestly as multi-corner / Monte-Carlo STA (not "incremental"). Test
  `test_corner_replay` proves each corner's GPU result matches its own CPU oracle AND
  differs from the base corner; the measured K=16 sweep beats a 24-core CPU 1.22×
  *including* the per-corner H2D delay upload. See lab-notebook E8.
