#!/usr/bin/env python3
"""bench.py — build the STA binary, sweep graph sizes, record MEASURED timings.

Honest by construction: the ``sta`` binary times the 1-core CPU reference, the fair
all-core CPU baseline (OpenMP), the per-launch GPU sweep, and the CUDA-graph replay
path (each min-of-N internally), and reports whether a CUDA device actually ran. On a
box with no GPU, staGpu falls back to the CPU reference and prints so; this script
records mode="cpu-only(fallback)" with empty GPU columns and NEVER invents a GPU
number. Speedups here are computed from the measured millisecond values, and the
fair reference is the MULTI-CORE CPU (not a single thread).

Usage:
    python3 bench/bench.py                 # configure+build, run default sweep
    python3 bench/bench.py --no-build      # reuse an existing build
    python3 bench/bench.py --reps 5        # min-of-N inside the binary (default 5)
    python3 bench/bench.py --csv out.csv   # where to write the CSV
    python3 bench/bench.py --sizes 128x512,400x4000

Output: a CSV (default bench/results.csv) plus a printed table on stdout.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SIZES = [(128, 512), (256, 1024), (400, 4000), (800, 8000)]

RE_GRAPH = re.compile(r"nodes=(\d+)\s+levels=(\d+)\s+arcs=(\d+)")
# One regex for every "CPU STA <N>-core" line; the 1-core and multi-core lines are
# then separated by their core count (findall — NOT search, which would stop at the
# 1-core line and mistake it for the multi-core baseline).
RE_CPU_CORE = re.compile(r"CPU STA\s+(\d+)-core\s+\(min of \d+\):\s*([-\d.eE+]+)\s*ms")
RE_WARM = re.compile(r"GPU STA warm \(min of \d+\):\s*([-\d.eE+]+)\s*ms")
RE_REPLAY = re.compile(r"GPU STA graph-replay \(min of \d+\):\s*([-\d.eE+]+)\s*ms")
RE_FALLBACK = re.compile(r"GPU STA: no CUDA device")


def sh(cmd, **kw):
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=True, **kw)


def configure_and_build(build_dir):
    os.makedirs(build_dir, exist_ok=True)
    sh(["cmake", "-S", REPO_ROOT, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"])
    sh(["cmake", "--build", build_dir, "--target", "sta", "-j"])
    exe = os.path.join(build_dir, "sta")
    if not os.path.exists(exe):
        sys.exit("error: expected binary not found at %s" % exe)
    return exe


def _ratio(num, den):
    return (num / den) if (den and den > 0) else None


def parse_run(text):
    g = RE_GRAPH.search(text)
    cores = {int(n): float(ms) for n, ms in RE_CPU_CORE.findall(text)}
    if not (g and 1 in cores and len(cores) >= 2):
        raise RuntimeError("could not parse sta output:\n" + text)
    ncores = max(cores)  # the multi-core baseline line
    row = {
        "nodes": int(g.group(1)), "levels": int(g.group(2)), "arcs": int(g.group(3)),
        "cpu1_ms": cores[1],
        "ncores": ncores, "cpuN_ms": cores[ncores],
        "warm_ms": None, "replay_ms": None,
        "warm_vs_N": None, "replay_vs_N": None, "replay_vs_1": None,
    }
    warm = RE_WARM.search(text)
    if warm:
        row["warm_ms"] = float(warm.group(1))
        row["warm_vs_N"] = _ratio(row["cpuN_ms"], row["warm_ms"])
        replay = RE_REPLAY.search(text)
        if replay:
            row["replay_ms"] = float(replay.group(1))
            row["replay_vs_N"] = _ratio(row["cpuN_ms"], row["replay_ms"])
            row["replay_vs_1"] = _ratio(row["cpu1_ms"], row["replay_ms"])
        row["mode"] = "gpu"
    elif RE_FALLBACK.search(text):
        row["mode"] = "cpu-only(fallback)"
    else:
        raise RuntimeError("could not classify GPU line:\n" + text)
    return row


def run_case(exe, levels, width, reps):
    out = subprocess.run([exe, str(levels), str(width), str(reps)],
                         check=True, capture_output=True, text=True).stdout
    row = parse_run(out)
    row["width"] = width
    return row


HEADER = ["levels", "width", "nodes", "arcs", "cpu1_ms", "ncores", "cpuN_ms",
          "warm_ms", "warm_vs_N", "replay_ms", "replay_vs_N", "replay_vs_1", "mode"]


def _cell(r, h):
    v = r.get(h)
    if v is None:
        return "-"
    if h.endswith("_ms"):
        return "%.3f" % v
    if h.startswith(("warm_vs", "replay_vs")):
        return "%.2fx" % v
    return str(v)


def print_table(rows):
    disp = [{h: _cell(r, h) for h in HEADER} for r in rows]
    widths = {h: len(h) for h in HEADER}
    for d in disp:
        for h in HEADER:
            widths[h] = max(widths[h], len(d[h]))

    def fmt(cells):
        return "  ".join(cells[h].rjust(widths[h]) for h in HEADER)

    print(fmt({h: h for h in HEADER}))
    print("  ".join("-" * widths[h] for h in HEADER))
    for d in disp:
        print(fmt(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=os.path.join(REPO_ROOT, "build"))
    ap.add_argument("--no-build", action="store_true", help="reuse existing build")
    ap.add_argument("--reps", type=int, default=5, help="min-of-N inside the binary (default 5)")
    ap.add_argument("--csv", default=os.path.join(REPO_ROOT, "bench", "results.csv"))
    ap.add_argument("--sizes", default=None,
                    help="comma list of LxW, e.g. 128x512,400x4000 (default sweep)")
    args = ap.parse_args()

    if args.sizes:
        sizes = []
        for tok in args.sizes.split(","):
            lvl, w = tok.lower().split("x")
            sizes.append((int(lvl), int(w)))
    else:
        sizes = DEFAULT_SIZES

    if args.no_build:
        exe = os.path.join(args.build_dir, "sta")
        if not os.path.exists(exe):
            sys.exit("error: --no-build but no binary at %s (run without --no-build)" % exe)
    else:
        exe = configure_and_build(args.build_dir)

    rows = [run_case(exe, lvl, w, args.reps) for lvl, w in sizes]
    any_gpu = any(r["mode"] == "gpu" for r in rows)

    with open(args.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=HEADER)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if r.get(k) is None else r[k]) for k in HEADER})

    print()
    print("STA benchmark  (measured, min-of-%d)" % args.reps)
    if any_gpu:
        print("device: CUDA GPU present — GPU times are MEASURED on-device.")
        print("        cpuN_ms = fair all-core CPU baseline (OpenMP); vs_N speedups are")
        print("        against it. warm = per-launch sweep; replay = CUDA-graph replay.")
    else:
        print("device: NO GPU — staGpu fell back to the CPU reference.")
        print("        GPU columns left blank on purpose (nothing measured to report).")
    print()
    print_table(rows)
    print()
    print("wrote %s" % args.csv)


if __name__ == "__main__":
    main()
