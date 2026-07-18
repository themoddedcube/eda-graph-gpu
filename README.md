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

## Measured on an H100 PCIe vs a fair 24-core CPU (CUDA 12.8, min-of-15)

Every GPU number is **measured on-device** and **bit-exact** against the CPU reference
(`maxAbsDiff == 0` at every size). The honest baseline is **all 24 CPU cores**
(`staCpuParallel`, the same level decomposition via OpenMP — bit-identical to the
serial reference), *not* one thread. `per-launch` is the naive level sweep;
`graph-replay` is the persistent-plan CUDA-graph path (iteration 2).

| nodes | CPU 1-core | **CPU 24-core** | GPU per-launch | GPU graph-replay | replay vs 24-core |
|---:|---:|---:|---:|---:|:--:|
| 262,144 | 5.08 ms | 3.53 ms | 3.95 ms | 1.82 ms | **1.9×** |
| 1,600,000 | 32.8 ms | 16.2 ms | 12.4 ms | 6.19 ms | **2.6×** |
| 6,400,000 | 131 ms | 27.6 ms | 32.5 ms | 15.9 ms | **1.7×** |

**Read it honestly — this is the number that survives review:**
1. Against a *single* core the GPU looks like 5–8×, but that is a strawman. Against a
   **fair all-core CPU** the honest win is **~1.7–2.6×**, and the naive per-launch GPU
   merely **ties** the 24-core CPU (0.85–1.31×) — so **the CUDA-graph optimization is
   what makes the GPU actually worth it**, not the raw offload.
2. The 24-core CPU baseline is memory-bandwidth-bound and **noisy on this shared cloud
   box** (the `replay vs 24-core` ratio moves ±0.4× run-to-run with CPU contention);
   treat these as approximate, not to three digits. The GPU times are stable.
3. The first GPU call pays a one-time CUDA context init (~210–310 ms), reported
   separately and amortized across a real re-evaluation loop — never hidden.

Reproduce: `python3 bench/bench.py` (writes `bench/results.csv`). Remaining
methodology gaps (real netlists vs the synthetic generator, a double-precision
ground truth, a changing-input replay, prior-work comparison) are tracked openly in
[`docs/red-team-review.md`](docs/red-team-review.md) and being worked through.

## Validated on real circuits (ISCAS-85)

Correctness isn't only shown on the synthetic generator: `readBench` ingests real
combinational netlists (a general DAG levelizer + `.bench` reader), and STA is
**bit-exact (oracle MATCH) on all 11 ISCAS-85 circuits** (c17…c7552), CPU and GPU —
`./build/profile bench/circuits` prints each design's level/degree profile beside the
generator's. Honest read: those real benchmarks are *small* (≤3.7k gates) so the CPU
wins there — the GPU regime is the large-scale graph — and the profile shows the
generator is more *uniform* (banded fanin, even level widths) than real circuits'
high-fanout hubs. Both facts are documented in [`docs/lab-notebook.md`](docs/lab-notebook.md) (E7);
closing the generator-realism gap is tracked next.

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
    (`staGpuPlanCreate`/`UpdateDelays`/`Run`/`Destroy`): **1.8–2.1× over per-launch,
    ~1.7–2.6× vs 24 cores**, bit-exact. The real workload is **multi-corner / Monte-Carlo
    re-evaluation**: capture the fixed topology once, then `UpdateDelays` + replay per
    corner — a measured **1.2× over a 24-core sweep** *including* the per-corner delay
    upload, with the result correctly *changing* each corner (not a re-run of one answer).
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
