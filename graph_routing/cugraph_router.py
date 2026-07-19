"""cugraph_router — route basic graph traversals to cuGraph primitives.

Build-vs-buy for basic graph ops: where cuGraph HAS a parallel primitive (BFS, WCC,
SSSP, PageRank) we route to it; where it does NOT (topological sort, DFS — both
inherently sequential and absent from cuGraph) we keep our own reference. Every
routed primitive is checked against the pure-Python oracle in `graphkit.py`.

Honest by construction: `available()` is False when cuGraph/cudf are not importable
(e.g. a CPU-only box / CI), and callers fall back to the reference — a routed result
is never fabricated.
"""


def available():
    try:
        import cudf  # noqa: F401
        import cugraph  # noqa: F401
        return True
    except Exception:
        return False


# Which cuGraph primitive each op maps to (None == cuGraph has no primitive).
MAPPING = [
    ("BFS reachability / logic cone", "cugraph.bfs", "buy"),
    ("Connected components (netlist partition)", "cugraph.weakly_connected_components", "buy"),
    ("Shortest path (weighted)", "cugraph.sssp", "buy"),
    ("Centrality / influence", "cugraph.pagerank", "buy"),
    ("Strongly-connected components", "cugraph.strongly_connected_components", "buy"),
    ("Topological sort / levelization", None, "BUILD (no cuGraph primitive) -> our levelizer"),
    ("DFS ordering", None, "BUILD (serial; no cuGraph primitive) -> CPU reference"),
]


def _graph(n, edges, directed, weighted=False):
    import cudf
    import cugraph

    src = [u for u, v, *_ in edges]
    dst = [v for u, v, *_ in edges]
    cols = {"src": src, "dst": dst}
    if weighted:
        cols["wgt"] = [w for _, _, w in edges]
    gdf = cudf.DataFrame(cols)
    G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(gdf, source="src", destination="dst",
                         edge_attr="wgt" if weighted else None)
    return G


def _by_vertex(df, value_col, n, default):
    """Align a cuGraph result DataFrame (vertex, value_col) to a dense [n] list."""
    out = [default] * n
    verts = df["vertex"].to_arrow().to_pylist()
    vals = df[value_col].to_arrow().to_pylist()
    for v, x in zip(verts, vals):
        if 0 <= v < n:
            out[v] = x
    return out


def bfs(n, edges, source):
    """cugraph.bfs -> hop distance per node (-1 if unreachable), aligned to node id."""
    import cugraph

    G = _graph(n, [(u, v) for u, v in edges], directed=True)
    df = cugraph.bfs(G, source)
    raw = _by_vertex(df, "distance", n, -1)
    # cuGraph marks unreachable with a large sentinel; normalize to -1 like the oracle.
    return [-1 if (d is None or d < 0 or d > n) else int(d) for d in raw]


def weakly_connected_components(n, edges):
    """cugraph.weakly_connected_components -> component label per node (0-based)."""
    import cugraph

    G = _graph(n, [(u, v) for u, v in edges], directed=False)
    df = cugraph.weakly_connected_components(G)
    labels = _by_vertex(df, "labels", n, None)  # None == absent (isolated node)
    # Dense-reindex cuGraph's labels; give each ISOLATED node (not in the edgelist,
    # so absent from cuGraph's result) its OWN singleton component — matching the
    # union-find convention. Real netlists have unconnected primary inputs, so this
    # is not a corner case.
    remap, out = {}, []
    for lab in labels:
        if lab is None:
            out.append(None)
        else:
            out.append(remap.setdefault(lab, len(remap)))
    k = len(remap)
    for i in range(n):
        if out[i] is None:
            out[i] = k
            k += 1
    return out, k


def sssp(n, weighted_edges, source):
    """cugraph.sssp -> shortest distance per node (inf if unreachable)."""
    import cugraph

    G = _graph(n, weighted_edges, directed=True, weighted=True)
    df = cugraph.sssp(G, source)
    raw = _by_vertex(df, "distance", n, float("inf"))
    return [float("inf") if (d is None or d > 1e30) else float(d) for d in raw]
