"""graphkit — zero-dependency reference graph traversals + a .bench loader.

These pure-Python implementations are the ORACLE that the cuGraph routing
(`route.py`) is checked against — the same discipline as the STA primitive: a
correct CPU reference, a GPU/library path, and an equivalence check between them.
No numpy/networkx required, so this always runs (even on a bare Python box / CI).
"""
from collections import deque


def load_bench(path):
    """Parse an ISCAS-85 `.bench` netlist.

    Returns (n, edges, names, inputs, outputs) where edges is a list of directed
    (fanin -> gate-output) pairs over integer node ids [0, n).
    """
    name2id = {}

    def gid(s):
        if s not in name2id:
            name2id[s] = len(name2id)
        return name2id[s]

    edges, inputs, outputs = [], [], []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].replace(" ", "").replace("\t", "").strip()
            if not line:
                continue
            if line.startswith("INPUT("):
                inputs.append(gid(line[6 : line.index(")")]))
            elif line.startswith("OUTPUT("):
                outputs.append(gid(line[7 : line.index(")")]))
            elif "=" in line:
                out, rhs = line.split("=", 1)
                o = gid(out)
                args = rhs[rhs.index("(") + 1 : rhs.index(")")]
                for a in args.split(","):
                    if a:
                        edges.append((gid(a), o))
    names = {v: k for k, v in name2id.items()}
    return len(name2id), edges, names, inputs, outputs


def adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    return adj


def topological_sort(n, edges):
    """Kahn's algorithm. Raises ValueError on a cycle (not a DAG)."""
    adj = adjacency(n, edges)
    indeg = [0] * n
    for _, v in edges:
        indeg[v] += 1
    q = deque(v for v in range(n) if indeg[v] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for w in adj[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(order) != n:
        raise ValueError("topological_sort: graph has a cycle")
    return order


def bfs_levels(n, edges, sources):
    """BFS distance (hop level) from `sources`; unreachable stays -1."""
    adj = adjacency(n, edges)
    dist = [-1] * n
    q = deque()
    for s in sources:
        if dist[s] == -1:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        for w in adj[u]:
            if dist[w] == -1:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


def sssp(n, weighted_edges, source):
    """Dijkstra single-source shortest path. weighted_edges: (u, v, w>=0).
    Returns dist[n] (float('inf') if unreachable)."""
    import heapq

    adj = [[] for _ in range(n)]
    for u, v, w in weighted_edges:
        adj[u].append((v, w))
    dist = [float("inf")] * n
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def dfs_preorder(n, edges, sources):
    """Iterative DFS preorder starting from each source in turn."""
    adj = adjacency(n, edges)
    seen = [False] * n
    order = []
    for s in sources:
        if seen[s]:
            continue
        stack = [s]
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            order.append(u)
            for w in reversed(adj[u]):
                if not seen[w]:
                    stack.append(w)
    return order


def weakly_connected_components(n, edges):
    """Union-find WCC. Returns (labels[n], count) with labels in [0, count)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    uniq, labels = {}, [0] * n
    for v in range(n):
        r = find(v)
        if r not in uniq:
            uniq[r] = len(uniq)
        labels[v] = uniq[r]
    return labels, len(uniq)


def same_partition(a, b):
    """True if two label vectors induce the SAME grouping (labels may differ)."""
    if len(a) != len(b):
        return False
    amap, bmap = {}, {}
    for x, y in zip(a, b):
        if amap.setdefault(x, y) != y or bmap.setdefault(y, x) != x:
            return False
    return True
