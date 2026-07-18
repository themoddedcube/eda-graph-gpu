# eda-graph-gpu

**Our own GPU primitives for the EDA graph workloads no vendor library covers.**

EDA is full of graph algorithms, but the ones that dominate its runtime — **static
timing analysis (STA)**, global/detailed **routing**, logic-synthesis DAG traversal —
have **no drop-in GPU library**. cuGraph is an analytics library (PageRank,
community, centrality); the CUDA Toolkit gives you cuFFT/cuBLAS/cuSPARSE for math,
not timing propagation. So for these workloads there is nothing to route *to* — you
have to build the primitive.

That's what this repo is: hand-written, CPU-oracle-checked GPU primitives for those
uncovered EDA graph workloads.

## Build vs. buy — the principle

Per-workload, decided by one question: *does a strong vendor primitive already exist?*

| Workload | Right answer |
|---|---|
| FFT (placement density), dense LA, sparse solve | **Use the library** — cuFFT / cuBLAS / cuSPARSE. Faster + verified; don't reinvent. |
| Netlist connectivity / reachability / cones | **Use cuGraph** — BFS / weakly-connected components genuinely fit. |
| **STA timing propagation, routing wavefronts** | **Build our own** — no vendor primitive exists. *This repo.* |

[CUDAadvisor](https://github.com/anandhkb/cuda_advisor) is the router that picks the
right tool per pattern: it routes to cuFFT/cuBLAS/cuSPARSE/cuGraph where they fit,
and to **these primitives** where they don't.

## Flagship: level-parallel STA

Static timing analysis propagates **arrival** times forward (max over fanin of
`arrival + delay`) and **required** times backward (min over fanout) across the
timing DAG. Nodes at the same topological **level** are mutually independent, so
each level is a fully data-parallel update — a textbook GPU **level-sweep**:

- `src/sta_cpu.cpp` — the CPU reference (the equivalence oracle) + a synthetic
  layered-DAG generator.
- `src/sta_gpu.cu` — the GPU primitive: one kernel launch per level, one thread per
  node, over CSR fanin/fanout. Serial across levels, parallel within.
- `src/main.cpp` — runs both, checks equivalence, reports the measured speedup.

### Honesty contract

- The GPU result is compared to the CPU reference with an **order-independent**
  comparator (STA outputs are deterministic value arrays — compare arrival/required
  within epsilon, never a path/order diff). A mismatch is a hard failure (exit 1).
- **No CUDA device → the GPU path falls back to the CPU reference and says so.** The
  same binary is correct everywhere and never fabricates a GPU result or speedup.
- Numbers reported by `main` are **measured**, not modeled.

## Measured on an H100 PCIe (CUDA 12.8, min-of-5)

Every GPU number below is **measured on-device** and **bit-exact** against the CPU
reference (`maxAbsDiff == 0` at every size). `warm` is the per-launch level sweep with
the CUDA context already live; `graph-replay` is the persistent-plan CUDA-graph path
(improvement iteration 2). Speedups are vs a single-thread CPU reference.

| nodes | arcs | CPU | GPU warm (per-launch) | GPU graph-replay | oracle |
|---:|---:|---:|---:|---:|:--:|
| 65,536 | 162,565 | 1.27 ms | 1.72 ms · 0.74× | **0.82 ms · 1.54×** | MATCH |
| 262,144 | 652,834 | 5.10 ms | 3.79 ms · 1.35× | **1.81 ms · 2.82×** | MATCH |
| 1,600,000 | 3,991,836 | 32.9 ms | 10.4 ms · 3.18× | **5.74 ms · 5.73×** | MATCH |
| 6,400,000 | 15,980,582 | 132 ms | 33.0 ms · 3.99× | **16.1 ms · 8.18×** | MATCH |

**Read it honestly:** (1) the *first* GPU call pays a one-time CUDA context init
(~210–310 ms) — amortized to nothing in a real STA loop that re-evaluates the same
graph, but a genuine cost we report separately, never hidden. (2) The naive per-launch
sweep is **launch-bound and loses on small graphs** (0.74× at 65k) — the CUDAadvisor
thesis on our own code. (3) The CUDA-graph plan collapses the ~2·levels per-level
launches into one replay, buying **1.8–2.1× over per-launch** and pushing the crossover
below 65k, scaling to **8.2× at 6.4M nodes**. Reproduce with `python3 bench/bench.py`.

## Build & run

```bash
make          # GPU build via nvcc (runs on GPU if present, else CPU fallback)
make cpu      # CPU-only build via g++ (no CUDA toolchain needed)
./sta 400 4000        # <levels> <nodes-per-level>   (default 400 x 4000)
```

The `.cu` compiles to a real `sm_90` fatbin (verify with `cuobjdump sta`); on a GPU
host `./sta` additionally runs the oracle check and prints the measured CPU→GPU
speedup.

## Roadmap

- **Stage 0:** level-parallel STA primitive — CPU reference + CUDA kernel + oracle +
  synthetic generator. ✅ builds (g++ + nvcc sm_90), CPU verified, **measured on an H100**.
- **Stage 1 (in progress):** tune the level sweep, each iteration measured on-device.
  - ✅ **iteration 1** — on-device period reduction (`cub::DeviceReduce::Max` over
    PO-masked arrivals) + fused slack; removes the mid-pipeline host round-trip.
  - ✅ **iteration 2** — CUDA-graph capture/replay via a persistent `StaGpuPlan`
    (`staGpuPlanCreate`/`Run`/`Destroy`): **1.8–2.1× over per-launch, up to 8.2× vs CPU**,
    bit-exact. The incremental-STA pattern: capture once, replay per evaluation.
  - ⏭ **iteration 3** — warp-per-node + `cub::WarpReduce` with degree bucketing
    (Warp-STAR) for the high-fanin load-imbalance case. Specified in `docs/research/`.
- **Stage 2:** real timing-graph ingestion (from a decoupled SoA netlist / a
  Bookshelf+library front-end) and a benchmark harness.
- **Stage 3:** connectivity/cones (route to **cuGraph** BFS/WCC — buy, don't build),
  then routing wavefronts (build).
- **Stage 4:** a `use_eda_graph_gpu` route in CUDAadvisor so it advises + emits calls
  to these primitives, measured-speedup gated, honest retain-CPU when they don't pay.

## Documentation

- [`docs/USAGE.md`](docs/USAGE.md) — external-user guide: the data model, the full API
  (`staCpu` / `staGpu` / the `StaGpuPlan` capture-replay lifecycle / `maxAbsDiff`), a
  verified hand-built example, build & integration, and reproduction.
- [`docs/lab-notebook.md`](docs/lab-notebook.md) — the engineering notebook: every
  measured result with its exact command + the oracle correctness check (append-only).
- [`docs/research/sta-optimization-roadmap.md`](docs/research/sta-optimization-roadmap.md)
  — cited SOTA survey and the prioritized optimization plan (iterations 1–2 done, 3 next).

## License

MIT (see `LICENSE`) — a deliberately clean, redistributable license for our own work.
