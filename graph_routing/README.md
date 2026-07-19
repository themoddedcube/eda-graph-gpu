# graph_routing — basic graph traversals → cuGraph (the "buy" half of the toolbox)

The STA primitive (this repo's flagship) is the **build** side: a GPU primitive for the
max-plus timing propagation no vendor library covers. This directory is the **buy**
side: the standard graph traversals that show up in EDA — reachability/cones,
connectivity, shortest paths — routed to **cuGraph** where it has a real parallel
primitive, and kept on our own reference where it doesn't. Same build-vs-buy thesis,
made concrete and oracle-checked.

## The mapping (oracle-checked)

| graph op (EDA use) | cuGraph primitive | decision |
|---|---|---|
| BFS reachability / logic cone | `cugraph.bfs` | **buy** ✅ validated |
| Connected components / netlist partition | `cugraph.weakly_connected_components` | **buy** ✅ validated |
| Shortest path (weighted) | `cugraph.sssp` | **buy** ✅ validated |
| Centrality / influence | `cugraph.pagerank` | buy (routed; not oracle-checked here) |
| Strongly-connected components | `cugraph.strongly_connected_components` | buy (routed) |
| **Topological sort / levelization** | **— none —** | **build** → our levelizer (`src/circuit.cpp`) |
| **DFS ordering** | **— none —** | **build** → CPU reference (`graphkit.dfs_preorder`) |

**The honest finding:** cuGraph is an *analytics* library — it provides the
parallel-friendly primitives (BFS, WCC/SCC, SSSP, PageRank) but has **no topological
sort and no DFS** (both inherently sequential). Those two — and topological sort is
exactly the levelization the STA primitive needs — have no cuGraph primitive to route
to, so we build them. That asymmetry is the whole point of a toolbox over a hammer.

## Validation (results from the test)

`test_routing.py` runs each routed primitive on the GPU and checks it **bit-for-bit
against the pure-Python oracle** (`graphkit.py`), on a synthetic graph and **all 11
ISCAS-85 circuits** (c17…c7552). Every case matches:

```
synthetic-7 : bfs MATCH  wcc MATCH (2 comps)  sssp MATCH
c17..c7552  : bfs MATCH  wcc MATCH            sssp MATCH   (all 11)
ROUTING OK — every cuGraph result matched the oracle.
```

Two real-netlist conventions were needed to make the oracle agree (documented in the
code): **isolated unconnected primary inputs** are their own WCC singletons (cuGraph
never sees nodes absent from the edgelist), and SSSP is run on a **simple** graph
(parallel arcs from a gate's repeated input are deduped so the min-weight path is
unambiguous). Neither is a cuGraph defect; both are graph-model choices.

## Files

- `graphkit.py` — zero-dependency reference traversals (BFS, DFS, topo sort, WCC,
  SSSP) + a `.bench` loader. The oracle; always runs, even with no GPU.
- `cugraph_router.py` — the cuGraph adapter (`available()`, `bfs`, `weakly_connected_components`,
  `sssp`, and the `MAPPING` table). Honest: `available()` is False without cuGraph and
  callers fall back to the reference — a routed result is never fabricated.
- `test_routing.py` — runs the mapping, prints results, oracle-checks, exits non-zero
  on any mismatch.

## Run

```bash
# needs cuGraph/cudf (RAPIDS). On this repo's dev box: .venv-rapids
.venv-rapids/bin/python graph_routing/test_routing.py [bench/circuits]
```

Without cuGraph installed the script still runs the reference oracle and reports the
routing as *wired, GPU validation skipped* — never a fake pass.
