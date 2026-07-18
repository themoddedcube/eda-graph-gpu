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

- **Stage 0 (this):** level-parallel STA primitive — CPU reference + CUDA kernel +
  oracle + synthetic generator. ✅ builds (g++ + nvcc sm_90), CPU verified.
- **Stage 1:** measure on a GPU host; tune the level-sweep (coalesced CSR, fanin
  load-balancing for high-degree nodes, fused arrival+slack).
- **Stage 2:** real timing-graph ingestion (from a decoupled SoA netlist / a
  Bookshelf+library front-end) and a benchmark harness.
- **Stage 3:** connectivity/cones (route to **cuGraph** BFS/WCC — buy, don't build),
  then routing wavefronts (build).
- **Stage 4:** a `use_eda_graph_gpu` route in CUDAadvisor so it advises + emits calls
  to these primitives, measured-speedup gated, honest retain-CPU when they don't pay.

## License

MIT (see `LICENSE`) — a deliberately clean, redistributable license for our own work.
