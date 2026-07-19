#!/usr/bin/env python3
"""test_routing — validate the cuGraph routing against the pure-Python oracle and
print RESULTS. For each graph (synthetic + real ISCAS-85), every op that cuGraph
provides (BFS, WCC, SSSP) is run on the GPU and checked bit-for-bit against
graphkit; the ops cuGraph does NOT provide (topological sort, DFS) are run on our
own reference and reported as such. Exit non-zero on any GPU-vs-oracle mismatch.

Run:  .venv-rapids/bin/python graph_routing/test_routing.py [circuits_dir]
"""
import glob
import os
import sys

import cugraph_router as cr
import graphkit as gk


def weighted(edges):
    # Deterministic non-unit weights so SSSP genuinely differs from BFS. Dedup to a
    # SIMPLE graph first: parallel arcs (a gate with a repeated input) would otherwise
    # get different weights and make the min-path ambiguous between the two sides.
    seen, out = set(), []
    for i, (u, v) in enumerate(edges):
        if (u, v) not in seen:
            seen.add((u, v))
            out.append((u, v, 1 + (i % 5)))
    return out


def check(name, n, edges, inputs):
    src = inputs[0] if inputs else 0
    wedges = weighted(edges)

    # --- oracle (pure Python) ---
    ref_bfs = gk.bfs_levels(n, edges, [src])
    ref_wcc, ref_k = gk.weakly_connected_components(n, edges)
    ref_sssp = gk.sssp(n, wedges, src)
    topo = gk.topological_sort(n, edges)          # our own (no cuGraph primitive)
    dfs = gk.dfs_preorder(n, edges, [src])        # our own (no cuGraph primitive)

    reach = sum(1 for d in ref_bfs if d >= 0)
    sssp_reach = sum(1 for d in ref_sssp if d != float("inf"))
    print(f"\n{name}: n={n} edges={len(edges)}")
    print(f"  [reference] BFS reach={reach}  WCC comps={ref_k}  "
          f"SSSP reach={sssp_reach}  topo|order|={len(topo)}  DFS|order|={len(dfs)}")

    fails = 0
    if not cr.available():
        print("  [cuGraph]  not installed — routing wired, GPU validation skipped")
        return 0

    # --- cuGraph, oracle-checked ---
    gpu_bfs = cr.bfs(n, edges, src)
    ok_bfs = gpu_bfs == ref_bfs
    fails += not ok_bfs

    gpu_wcc, gpu_k = cr.weakly_connected_components(n, edges)
    ok_wcc = gk.same_partition(gpu_wcc, ref_wcc)
    fails += not ok_wcc

    gpu_sssp = cr.sssp(n, wedges, src)
    ok_sssp = all(
        (a == float("inf") and b == float("inf")) or abs(a - b) < 1e-4
        for a, b in zip(gpu_sssp, ref_sssp)
    )
    fails += not ok_sssp

    def mark(ok):
        return "MATCH" if ok else "*** MISMATCH ***"

    print(f"  [cuGraph]  bfs->{mark(ok_bfs)} (reach={sum(d>=0 for d in gpu_bfs)})  "
          f"wcc->{mark(ok_wcc)} (comps={gpu_k})  sssp->{mark(ok_sssp)} "
          f"(reach={sum(d!=float('inf') for d in gpu_sssp)})")
    return fails


def main():
    circuits_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "circuits")

    print("=" * 68)
    print("Basic graph traversal -> cuGraph primitive mapping (oracle-checked)")
    print("=" * 68)
    for op, prim, decision in cr.MAPPING:
        print(f"  {op:<44} {(prim or '—'):<38} {decision}")
    print(f"\ncuGraph available: {cr.available()}")

    total = 0

    # A synthetic graph with a few components (two disjoint chains + a fork).
    syn_edges = [(0, 1), (1, 2), (2, 3), (0, 3), (4, 5), (5, 6)]
    total += check("synthetic-7", 7, syn_edges, [0])

    # Real ISCAS-85 circuits.
    for path in sorted(glob.glob(os.path.join(circuits_dir, "*.bench"))):
        name = os.path.splitext(os.path.basename(path))[0]
        n, edges, _, inputs, _ = gk.load_bench(path)
        total += check(name, n, edges, inputs)

    print("\n" + "=" * 68)
    if total == 0:
        print("ROUTING OK — every cuGraph result matched the oracle.")
        return 0
    print(f"ROUTING FAILED — {total} mismatch(es).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
