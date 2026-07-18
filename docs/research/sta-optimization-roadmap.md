# GPU STA optimization roadmap — for our level-parallel primitive

> **Implementation status (measured on an H100 PCIe, CUDA 12.8):**
> **#1 on-device period reduction + fused slack — DONE**, bit-exact.
> **#2 CUDA-graph capture/replay (`StaGpuPlan`) — DONE**, measured **1.8–2.1× over the
> per-launch path, up to 8.2× vs CPU** at 6.4M nodes, bit-exact (see README table).
> **#3 warp-per-node + degree bucketing — NOT STARTED** (specified in §3 below).
> The ESTIMATED figures below were written *before* measurement — keep them labeled as
> estimates; the README carries the measured numbers.

Scope: research the state of the art in GPU-accelerated static timing analysis (STA)
and level-parallel DAG propagation, then turn it into a concrete, prioritized plan for
**our** primitive in `src/sta_gpu.cu` / `src/sta_cpu.cpp`.

Honesty labels used throughout:

- **MEASURED (paper)** — a number a cited paper actually measured. Note that every one
  of these is on a *different* workload/baseline/GPU than ours; it bounds what the
  technique *can* buy, it is **not** a prediction for our kernel.
- **ESTIMATED (ours)** — a first-order model of the effect on *our* code. Never
  measured. To become MEASURED it needs the harness in `src/main.cpp` run on a GPU host.

Our code as it stands (evidence, `path:line`):

- One kernel launch **per topological level**, per direction: forward loop
  `src/sta_gpu.cu:76-82`, backward loop `src/sta_gpu.cu:95-101`. Default run is
  `400 x 4000` (`README.md:56`), i.e. ~400 launches each way, ~800 total.
- **One thread per node**, serial `for` loop over that node's fanin:
  `arrivalKernel` `src/sta_gpu.cu:23-34` (reduction loop `:32`), `requiredKernel`
  `src/sta_gpu.cu:37-49` (loop `:47`).
- Fanin gather `arrival[finFrom[p]]` (`src/sta_gpu.cu:32`) is a **data-dependent,
  uncoalesced** load; `finFrom`/`finDelay` themselves are contiguous per node.
- Variable fanin degree by construction: `1 + rng()%maxFaninPerNode`
  (`src/sta_cpu.cpp:95`) → intra-warp load imbalance.
- **Mid-pipeline host stall**: after the forward sweep we `cudaDeviceSynchronize`,
  copy the whole `arrival` array D2H, and reduce `period` in a host loop over primary
  outputs (`src/sta_gpu.cu:83-92`) before relaunching the backward sweep. `slack` is
  also computed on the host (`src/sta_gpu.cu:108-109`).
- All graph arrays are uploaded once up front (`src/sta_gpu.cu:66-72`) — good; but H2D
  is not overlapped with compute and memory is pageable (`upload()` `:51-56`).
- `TPB = 256` fixed (`src/sta_gpu.cu:74`); tiny levels waste the tail block.

---

## 1. State of the art (with citations)

### 1.1 Block-based (what we do) vs path-based STA

STA has two traversal families. **Block-based** (a.k.a. graph-based, PERT-like)
computes an arrival/required value *per node* by a forward+backward sweep over the
timing DAG — fast, complete, no path enumeration, but pessimistic. **Path-based**
analysis (PBA) sums delays along *specific* critical paths to remove that pessimism;
it is far more accurate but has exponential worst-case path count. Our primitive is
squarely **block-based** — the `max`-over-fanin forward pass and `min`-over-fanout
backward pass in `staCpu`/`arrivalKernel`/`requiredKernel` are exactly the PERT
traversal. (Statistical STA survey, Wikipedia:
https://en.wikipedia.org/wiki/Statistical_static_timing_analysis ; IBM
"First-Order Incremental Block-Based Statistical Timing Analysis":
https://people.eecs.berkeley.edu/~alanmi/research/timing/papers/sta_ibm.pdf )

**Incremental STA** re-times only the fan-in/fan-out cone affected by a netlist edit
rather than the whole graph — the dominant cost in the inner loop of place-and-route
and physical synthesis. OpenTimer v2 is built around a task-based incremental engine.
(Huang et al., OpenTimer / Taskflow line of work, below.)

### 1.2 GPU block-based STA — the directly-comparable prior art

- **Guo, Huang, Lin — "GPU-Accelerated Static Timing Analysis," ICCAD 2020.**
  https://dl.acm.org/doi/10.1145/3400302.3415631 (PDF:
  https://tsung-wei-huang.github.io/papers/iccad20-gpusta.pdf ). Builds GPU-efficient
  data structures and kernels for **levelization, delay calculation, and graph update**,
  inside a task-based **CPU-GPU heterogeneous** framework that overlaps kernel compute
  with data movement and falls back to CPU when parallelism is scarce (incremental
  mode). **MEASURED (paper): up to 3.69x** on a 1.6M-gate / 1.6M-net design vs
  OpenTimer on 40 CPUs; "one GPU can run faster than OpenTimer of 40 CPUs." This is the
  closest published analogue to our level-sweep and the single best reference to mine.

- **Warp-STAR — "High-performance, Differentiable GPU-Accelerated STA through
  Warp-oriented Parallel Orchestration."** https://arxiv.org/html/2603.28381 . Directly
  attacks the load-imbalance problem our thread-per-node kernel has. Prior GPU STA
  mapped **one net to one thread**; because nets have different pin counts, "threads
  within a warp operate in lockstep" and diverge. Warp-STAR instead processes "the pins
  within a net ... by the unit of warp," distributing one node's fanin across a warp,
  and does the reduction with a **shared-memory tree reduction guarded by
  `__syncwarp()`** (their Algorithm 1, lines 24-28) to avoid atomics. It still launches
  **one kernel per topological level** ("each level is described by a CUDA kernel
  function") — no persistent kernel / CUDA graph / grid-sync. **MEASURED (paper): 2.4x
  over the previous GPU-Timer and 162x over OpenTimer (CPU)**; superblue1 5095 ms (CPU)
  → 57 ms (GPU-Timer) → 25 ms (Warp-STAR). They also **fuse operations** to cut
  differentiable-STA gradient overhead from 33% to 16%. Caveat they report themselves:
  a fancier "compressed" load-balancing scheme (CTE) did **not** beat the simpler
  warp-per-net one — the overhead ate the theoretical gain. Relevant honesty check for
  us: don't over-engineer the load balancer.

- **"Accelerating STA using CPU-GPU Heterogeneous Parallelism."**
  https://www.researchgate.net/publication/371589918 — decomposes STA into
  CPU-GPU dependent tasks with overlapped kernel compute and data processing; same
  research line, reinforces the heterogeneous-overlap idea.

### 1.3 GPU path-based / current-source-model STA (future stages, not stage 1)

- **Guo, Huang, Lin, Wong — "GPU-accelerated Path-based Timing Analysis," DAC 2021.**
  https://dl.acm.org/doi/10.1109/DAC18074.2021.9586316 (PDF:
  https://yibolin.com/publications/papers/TIMER_TopK_DAC2021_Guo.pdf ). Compact data
  structures + efficient kernels + GPU preprocessing of path constraints. **MEASURED
  (paper): 543x on leon2 (1.6M gates), 172x on leon3mp (1.2M), 304x on netcard (1.5M)**
  vs a CPU baseline.
- **"A GPU-Accelerated Framework for Path-Based Timing Analysis," TCAD 2023.**
  https://dl.acm.org/doi/10.1109/TCAD.2023.3272274 (journal extension of the above).
- **GCS-Timer — "GPU-Accelerated Current Source Model Based STA," DAC 2024.**
  https://dl.acm.org/doi/10.1145/3649329.3655983 — moves beyond lumped arc delays to
  CSM waveform propagation on GPU. Relevant only once we add a real delay model.
- **INSTA — "Ultra-Fast, Differentiable, Statistical STA," DAC 2025 (NVIDIA).**
  https://research.nvidia.com/labs/electronic-design-automation/papers/yichen_INSTA_dac25.pdf
  — differentiable/statistical STA on GPU, the direction the field is heading.

### 1.4 Kernel-level technique base (transferable from SpMV / graph-propagation)

Our forward pass is structurally a **max-plus SpMV**: for each row (node) reduce over
its nonzeros (fanin arcs). All the CSR-SpMV load-balancing literature applies.

- **Load imbalance from variable row length.** "CSR-based SpMV on GPUs has poor
  performance due to irregular memory access, load imbalance, and reduced parallelism."
  Thread-per-row is simple but imbalanced when degrees are skewed; warp-per-row balances
  within a warp but wastes lanes on short rows. Fixes: **row bundling / reorder by
  degree**, **nonzero-split (COO / merge-based)**, and **hybrid formats**.
  - Anzt et al., "Load-balancing Sparse Matrix Vector Product Kernels on GPUs," ACM
    TOPC 2020: https://dl.acm.org/doi/fullHtml/10.1145/3380930
  - LightSpMV (dynamic warp/vector assignment by row length):
    https://www.researchgate.net/publication/276411706
  - Bell & Garland, "Implementing SpMV on Throughput-Oriented Processors," SC09 (the
    origin of CSR-vector / hybrid ELL+COO):
    https://www.nvidia.com/docs/io/77944/sc09-spmv-throughput.pdf
- **Warp-level reduction primitives.** `cub::WarpReduce` / `__shfl_down_sync` do a
  register-only max over a warp with no shared memory and no atomics — the natural tool
  for the fanin reduction when a warp cooperates on one node. (CUB is a CUDA-Toolkit
  library, not a third-party dep.)
- **Kernel-launch-overhead amortization** (our biggest structural cost — ~800 launches):
  - **CUDA Graphs**: record a DAG of GPU ops once, replay with a single
    `cudaGraphLaunch`. Each ordinary launch costs ~5-10 µs of host/driver time; graphs
    remove that per-op and the inter-kernel idle gaps. **MEASURED (paper): a 3-kernel /
    100-iteration pipeline dropped from ~1.00 ms to ~0.75 ms (~25%)** by collapsing 300
    launches into 100 replays. (Kernel-batching-with-CUDA-Graphs study:
    https://arxiv.org/html/2501.09398v1 ; NVIDIA CUDA Graphs docs:
    https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html )
  - **Persistent kernel + cooperative-groups grid sync**: launch once, keep blocks
    resident, and use a **grid-wide barrier** (`this_grid().sync()` via
    `cudaLaunchCooperativeKernel`) between levels instead of relaunching. "As a
    persistent kernel, it pays the launch cost exactly once." Constraint: a cooperative
    launch's grid may be **no larger than the number of blocks that fit concurrently on
    the device** — so this works when one level fits resident, and needs a grid-stride
    fallback for wide levels. (NVIDIA "Cooperative Groups":
    https://developer.nvidia.com/blog/cooperative-groups/ ; CUDA Programming Guide,
    Cooperative Groups:
    https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html )

### 1.5 Build-vs-buy: is there a library to route to? (honest read)

Consistent with `README.md:5-27`: **no vendor library does max-plus timing propagation
over a topological DAG.** cuGraph is analytics (BFS/PageRank/WCC), cuSPARSE does
`+`/`*` SpMV over a semiring you can't redefine to `max`/`+`. So the *primitive* stays
hand-built — correctly. But three **CUB (CUDA Toolkit) building blocks** genuinely fit
*parts* of it and should be bought, not hand-rolled:

- `cub::DeviceReduce::Max` for the **period** reduction over PO arrivals (replaces the
  host loop at `src/sta_gpu.cu:89-91`).
- `cub::WarpReduce<float>::Reduce(..., cub::Max())` for the **fanin reduction** in the
  warp-cooperative kernel.
- `cub::DeviceScan::ExclusiveSum` to build CSR `finStart`/`foutStart` prefix sums if we
  ever move graph construction to the device.

And **levelization itself** (currently trivial because the generator lays nodes out by
level, `src/sta_cpu.cpp:71-79`) is a topological sort — for *ingested* real netlists
(Stage 2) that is a genuine **cuGraph BFS/topo** fit, matching the repo's Stage 3 plan
(`README.md:70-74`). Buy that; don't hand-write a GPU topo sort.

---

## 2. Prioritized optimization table (for our code)

Impact is for the default `400 x 4000` synthetic run and is **ESTIMATED (ours)** unless
tagged otherwise. "MEASURED (paper)" columns cite what the *technique* bought in its
source paper on a *different* workload — an upper reference, not a prediction for us.

| # | Technique | File / function it changes | Expected impact | Difficulty |
|---|-----------|----------------------------|-----------------|------------|
| P1 | **CUDA Graph capture of the level sweeps** — record the whole forward (and backward) chain of per-level launches once, replay with one `cudaGraphLaunch`. Kills ~800 launches' host overhead + inter-kernel gaps. | `runGpu`, `src/sta_gpu.cu:76-82` & `:95-101` | ESTIMATED (ours): removes ~800 × ~5-10 µs ≈ **4-8 ms** of pure launch overhead; dominant when levels are thin (4000 nodes = ~16 blocks, kernel < launch cost). MEASURED (paper): ~25% pipeline latency on a 3-kernel loop (arxiv 2501.09398). | **Low–Med** |
| P2 | **Warp/sub-warp-per-node + warp-shuffle reduction**, with **degree bucketing** within a level (short-fanin nodes → thread-per-node; high-fanin → warp-per-node). Fixes intra-warp load imbalance from `1..maxFanin` degrees. | `arrivalKernel`/`requiredKernel` `src/sta_gpu.cu:23-49`; add a bucket split in `runGpu` | ESTIMATED (ours): removes warp-divergence stalls on skewed levels; larger as `maxFaninPerNode` grows. MEASURED (paper): **2.4x** for warp-per-net vs thread-per-net GPU-Timer (Warp-STAR). | **Med–High** |
| P3 | **Eliminate the mid-pipeline host stall**: compute `period` on-device with `cub::DeviceReduce::Max` over PO arrivals and fuse `slack` into a device kernel, so forward+backward+slack run as one uninterrupted stream (and become a single CUDA-graph, unblocking P1 across *both* sweeps). | `src/sta_gpu.cu:83-92`, `:108-109` | ESTIMATED (ours): removes one full-array D2H + host reduce + resync between sweeps; also a prerequisite that makes P1 span both directions. | **Low–Med** |
| P4 | **Persistent kernel + cooperative-groups grid sync** as an alternative to P1: one launch, `grid.sync()` between levels. | new kernel replacing the level loop `src/sta_gpu.cu:76-101` | ESTIMATED (ours): removes launch overhead like P1 with no replay setup, but needs a grid-stride fallback when a level exceeds resident-block capacity. MEASURED (paper): "pays launch cost once" (NVIDIA coop-groups). | **High** |
| P5 | **Coalesce / prefetch CSR & drop redundant loads**: read `finStart[v]`/`finStart[v+1]` once; consider a per-node `(start,count)` int2 so the bound load is one coalesced access; `__ldg`/`const __restrict__` on read-only arrays. | `arrivalKernel`/`requiredKernel` `src/sta_gpu.cu:29,32,44,47` | ESTIMATED (ours): modest (single-digit %); the `arrival[finFrom[p]]` gather stays irregular regardless. | **Low** |
| P6 | **Overlap H2D upload with compute + pinned memory**: `cudaMallocHost` + `cudaMemcpyAsync` on a stream so uploads (`src/sta_gpu.cu:51-72`) overlap the first levels. | `upload`, `runGpu` `src/sta_gpu.cu:51-72` | ESTIMATED (ours): hides upload latency; matters most for large ingested graphs, not the tiny synthetic one. | **Low–Med** |
| P7 | **Tune block size / fuse tail** — `TPB` (`src/sta_gpu.cu:74`) is fixed at 256; thin levels waste the last block. Autotune or use grid-stride so few blocks cover any level width. | `runGpu` `src/sta_gpu.cu:74-101` | ESTIMATED (ours): small; mostly relevant with P4. | **Low** |
| P8 | **(Stage 2+) Incremental STA**: re-time only the affected fan-in/fan-out cone on netlist edits, CPU when parallelism is scarce, GPU when massive — the ICCAD'20 adaptive model. | new API on top of `staGpu` | MEASURED (paper): the adaptive CPU-GPU model underlies the 3.69x whole-flow result (ICCAD'20). | **High** |
| P9 | **(Stage 3) Route levelization to cuGraph** for *ingested* real netlists (topo sort / BFS); keep it hand-rolled only for the synthetic generator. | new ingestion path (Stage 2/3) | Honest build-vs-buy; matches `README.md:70-74`. | **Med** |

Do **P1 + P3 first** (low risk, attack the dominant structural cost — launch overhead
and the mid-pipeline stall). P2 is the biggest single-kernel win but higher risk; gate
it behind the `main.cpp` measured-speedup harness, and heed Warp-STAR's own warning
that an over-clever balancer (their CTE) lost to the simple one.

---

## 3. Top-3 changes, fully specified

All three are validated the same way: `src/main.cpp` already runs CPU + GPU and checks
equivalence with `maxAbsDiff` (`src/sta.h:31`, `src/sta_cpu.cpp:56-64`); every change
below must keep `maxAbsDiff` at ~0 (float epsilon) and only then report a speedup. Per
the repo contract, no `nvcc` compile / GPU run ⇒ it stays `manual_review`, not a pass.

### 3.1 (P3) On-device period reduction + fused slack — removes the mid-pipeline stall

**Why first:** it is low-risk, and it is the *enabler* for P1 spanning both sweeps.
Today the sequence is: forward launches → `cudaDeviceSynchronize` → D2H whole `arrival`
→ host loop for `period` → backward launches → D2H `required` → host loop for `slack`
(`src/sta_gpu.cu:83-109`). The host `period` reduce forces a full sync and a
host round-trip *between* the two GPU phases, which also prevents capturing both sweeps
in one graph.

Replace the host period loop (`src/sta_gpu.cu:89-91`) with a device reduce over PO
arrivals. We already know PIs/POs by CSR emptiness (`isPrimaryOutput` = `foutStart[v]==
foutStart[v+1]`, `timing_graph.h:41`). Precompute a device array of PO node ids once
(host side, from `foutStart`), then:

```cpp
// One-time (host): poNodes = { v : foutStart[v]==foutStart[v+1] }; upload to dPoNodes.
// Device: gather PO arrivals then reduce-max with CUB (no host round-trip).
__global__ void gatherPoArrival(const int* poNodes, int nPo,
                                const float* arrival, float* poArr) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nPo) poArr[i] = arrival[poNodes[i]];
}
// cub::DeviceReduce::Max(d_temp, bytes, dPoArr, dPeriod, nPo, stream);
// dPeriod (device float) feeds requiredKernel directly — no D2H of `arrival` here.
```

Pass `dPeriod` (device pointer) into `requiredKernel` instead of the host `period`
float (`requiredKernel` signature `src/sta_gpu.cu:37-40`, launch `:99-100`); dereference
it inside the kernel for the PO seed (`src/sta_gpu.cu:45`). Fuse slack into the tail:

```cpp
__global__ void slackKernel(int n, const float* arrival, const float* required,
                            float* slack) {
    int v = blockIdx.x*blockDim.x + threadIdx.x;
    if (v < n) slack[v] = required[v] - arrival[v];   // replaces host loop :108-109
}
```

Now the only host↔device traffic between phases is gone; `arrival`/`required`/`slack`
are copied back once at the very end. Correctness is identical (same max/min values),
so `maxAbsDiff` stays ~0.

### 3.2 (P1) CUDA Graph capture of the level sweeps

**Why:** the default graph has ~400 levels; each level's kernel touches only ~4000
nodes (~16 blocks at `TPB=256`) — kernel runtime is comparable to the ~5-10 µs launch
cost, so ~800 launches are largely host overhead and inter-kernel gaps. Record the
whole sweep once, replay with a single call. With 3.1 done, both sweeps + slack sit in
one stream with no host interruption, so one graph covers the entire GPU phase.

Use **stream capture** (no kernel rewrite — the existing per-level loop is captured
verbatim):

```cpp
cudaStream_t s; cudaStreamCreate(&s);
cudaGraph_t graph; cudaGraphExec_t exec;

cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
for (int L = 0; L < g.numLevels; ++L) {                 // was src/sta_gpu.cu:76-82
    int lo = g.levelStart[L], hi = g.levelStart[L+1], count = hi-lo;
    if (count <= 0) continue;
    arrivalKernel<<<(count+TPB-1)/TPB, TPB, 0, s>>>(     // note the stream arg
        lo, hi, dLevelNodes, dFinStart, dFinFrom, dFinDelay, dArrival);
}
gatherPoArrival<<<...,0,s>>>(...);                       // from 3.1
// cub::DeviceReduce::Max(..., s);
for (int L = g.numLevels-1; L >= 0; --L) {               // was src/sta_gpu.cu:95-101
    int lo = g.levelStart[L], hi = g.levelStart[L+1], count = hi-lo;
    if (count <= 0) continue;
    requiredKernel<<<(count+TPB-1)/TPB, TPB, 0, s>>>(
        lo, hi, dLevelNodes, dFoutStart, dFoutTo, dFoutDelay, dPeriod, dRequired);
}
slackKernel<<<...,0,s>>>(...);
cudaStreamEndCapture(s, &graph);
cudaGraphInstantiate(&exec, graph, 0);

cudaGraphLaunch(exec, s);                                // one host call for ~800 kernels
cudaStreamSynchronize(s);
```

Kernel bodies are unchanged, so equivalence is preserved by construction; the only
behavioral change is *when* the launches hit the driver. The graph is topology-static
(same levels every run) so it can even be instantiated once and replayed across many
timing updates — directly the amortization the kernel-batching study measured (~25%,
arxiv 2501.09398). The `-opt`/harness in `main.cpp` should time *replay*, not the
one-time `cudaGraphInstantiate`, and report it honestly as such.

### 3.3 (P2) Warp/sub-warp-per-node with warp-shuffle reduction + degree bucketing

**Why:** the reduction loop `for (p=s; p<e; ++p) a = fmaxf(a, arrival[finFrom[p]]+
finDelay[p])` (`src/sta_gpu.cu:32`) runs on **one thread**, and fanin degree varies
`1..maxFaninPerNode` (`src/sta_cpu.cpp:95`). In a 32-lane warp, one node with 8 fanins
stalls 31 lanes that have 1 fanin each — the exact intra-warp imbalance Warp-STAR fixed
by moving to warp granularity (2.4x, §1.2).

**Degree bucketing (do the simple, measured-to-win version).** Warp-STAR found the
*simple* warp-per-node scheme beat their cleverer compressed one — so split each level
into two passes by a degree threshold `D` (e.g. 4) instead of a fully dynamic balancer:

- **low bucket** (`degree ≤ D`): keep today's thread-per-node kernel
  (`src/sta_gpu.cu:23-34`) — it is already efficient for short rows.
- **high bucket** (`degree > D`): a warp-cooperative kernel, one warp per node, lanes
  stride the fanin and reduce with a warp shuffle.

Bucket membership is static per graph (degrees don't change), so compute two index
lists per level once on the host from `finStart` and upload them; no device sort needed.

Warp-cooperative arrival kernel (uses `cub::WarpReduce`, a CUDA-Toolkit primitive —
buy, don't hand-roll the shuffle tree):

```cpp
// Grid: one warp per high-degree node in this level. warpId picks the node.
__global__ void arrivalKernelWarp(int lstart, int lend, const int* highNodes,
                                  const int* finStart, const int* finFrom,
                                  const float* finDelay, float* arrival) {
    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp[WARPS_PER_BLOCK];
    int warp = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int slot = lstart + warp;
    if (slot >= lend) return;
    int v = highNodes[slot];
    int s = finStart[v], e = finStart[v+1];
    float a = -3.402823e38f;                              // -FLT_MAX, matches :31
    for (int p = s + lane; p < e; p += 32)               // lanes stride the fanin
        a = fmaxf(a, arrival[finFrom[p]] + finDelay[p]);
    float amax = WarpReduce(temp[threadIdx.x>>5]).Reduce(a, cub::Max());
    if (lane == 0) arrival[v] = amax;                    // one writer, no atomics
}
```

The backward `requiredKernel` gets the mirror treatment (`min` over fanout via
`cub::Min`, `foutStart`/`foutTo`/`foutDelay`, seed from `dPeriod` for POs). Numerically
identical `max`/`min` over the same arc set ⇒ `maxAbsDiff` unchanged. `D` and
`WARPS_PER_BLOCK` are tuning knobs to sweep in the `main.cpp` harness; keep the simple
two-bucket split unless a measurement (not a model) justifies more, per Warp-STAR's own
negative result on over-engineering.

---

### Validation & honesty reminders (repo contract)

- Every change keeps the `maxAbsDiff` oracle check (`src/main.cpp`, `src/sta.h:31`) at
  float epsilon; a mismatch is a hard fail, never a relaxed gate.
- Speedups are only real when `main` measures them on a GPU host after a real `nvcc`
  compile; on a CPU-only host the primitive still falls back honestly
  (`staGpu`/`runGpu` `src/sta_gpu.cu:119-129`) and reports no GPU number.
- The paper speedups in §1-2 are MEASURED **in those papers' settings** — treat them as
  the ceiling a technique reached elsewhere, not as our result until our harness prints
  it.

## Sources

- GPU-Accelerated STA, ICCAD 2020: https://dl.acm.org/doi/10.1145/3400302.3415631 · PDF https://tsung-wei-huang.github.io/papers/iccad20-gpusta.pdf
- Warp-STAR: https://arxiv.org/html/2603.28381
- Accelerating STA using CPU-GPU Heterogeneous Parallelism: https://www.researchgate.net/publication/371589918
- GPU-accelerated Path-Based Timing Analysis, DAC 2021: https://dl.acm.org/doi/10.1109/DAC18074.2021.9586316 · PDF https://yibolin.com/publications/papers/TIMER_TopK_DAC2021_Guo.pdf
- GPU-Accelerated Framework for Path-Based Timing Analysis, TCAD 2023: https://dl.acm.org/doi/10.1109/TCAD.2023.3272274
- GCS-Timer, DAC 2024: https://dl.acm.org/doi/10.1145/3649329.3655983
- INSTA, DAC 2025 (NVIDIA): https://research.nvidia.com/labs/electronic-design-automation/papers/yichen_INSTA_dac25.pdf
- Load-balancing SpMV kernels on GPUs, ACM TOPC 2020: https://dl.acm.org/doi/fullHtml/10.1145/3380930
- LightSpMV: https://www.researchgate.net/publication/276411706
- Bell & Garland, SpMV on throughput processors, SC09: https://www.nvidia.com/docs/io/77944/sc09-spmv-throughput.pdf
- CUDA Graphs kernel batching study: https://arxiv.org/html/2501.09398v1 · NVIDIA CUDA Graphs docs: https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html
- NVIDIA Cooperative Groups: https://developer.nvidia.com/blog/cooperative-groups/ · CUDA Programming Guide (Cooperative Groups): https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
- Statistical STA (block- vs path-based) survey: https://en.wikipedia.org/wiki/Statistical_static_timing_analysis · IBM incremental block-based SSTA: https://people.eecs.berkeley.edu/~alanmi/research/timing/papers/sta_ibm.pdf
