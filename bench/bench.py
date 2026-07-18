#!/usr/bin/env python3
"""bench.py — build the STA binary, sweep graph sizes, record MEASURED timings.

Honest by construction: the ``sta`` binary times staCpu and staGpu itself and
reports whether a CUDA device actually ran. On a box with no GPU (this dev box,
GitHub-hosted CI) staGpu falls back to the CPU reference and prints so; this
script records mode="cpu-only(fallback)" with an empty GPU column and NEVER
invents a GPU time or speedup. When a device is present it parses the measured
GPU time + speedup straight from the binary's output.

Usage:
    python3 bench/bench.py                 # configure+build, run default sweep
    python3 bench/bench.py --no-build      # reuse an existing build
    python3 bench/bench.py --reps 5        # min-of-N timing (default 3)
    python3 bench/bench.py --csv out.csv   # where to write the CSV
    python3 bench/bench.py --sizes 32x128,64x256

Output: a CSV (default bench/results.csv) plus a printed table on stdout.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (levels, width) sweep. STA is O(V+E) so even the big one finishes fast on CPU.
DEFAULT_SIZES = [(16, 64), (32, 128), (64, 256), (128, 512), (256, 1024)]

RE_GRAPH = re.compile(r"nodes=(\d+)\s+levels=(\d+)\s+arcs=(\d+)")
RE_PERIOD = re.compile(r"period\):\s*([-\d.eE+]+)")
RE_CPU = re.compile(r"CPU STA:\s*([-\d.eE+]+)\s*ms")
RE_GPU = re.compile(r"GPU STA:\s*([-\d.eE+]+)\s*ms\s*\(([-\d.eE+]+)x\)")
RE_GPU_FALLBACK = re.compile(r"GPU STA:\s*no CUDA device")


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


def parse_run(text):
    g = RE_GRAPH.search(text)
    p = RE_PERIOD.search(text)
    cpu = RE_CPU.search(text)
    if not (g and p and cpu):
        raise RuntimeError("could not parse sta output:\n" + text)
    row = {
        "nodes": int(g.group(1)),
        "levels": int(g.group(2)),
        "arcs": int(g.group(3)),
        "period": float(p.group(1)),
        "cpu_ms": float(cpu.group(1)),
    }
    gpu = RE_GPU.search(text)
    if gpu:
        row["gpu_ms"] = float(gpu.group(1))
        row["speedup"] = float(gpu.group(2))
        row["mode"] = "gpu"
    elif RE_GPU_FALLBACK.search(text):
        row["gpu_ms"] = None
        row["speedup"] = None
        row["mode"] = "cpu-only(fallback)"
    else:
        raise RuntimeError("could not classify GPU line:\n" + text)
    return row


def run_case(exe, levels, width, reps):
    best = None
    for _ in range(reps):
        out = subprocess.run([exe, str(levels), str(width)],
                             check=True, capture_output=True, text=True).stdout
        row = parse_run(out)
        # keep the fastest (min) CPU time; carry its GPU fields along
        if best is None or row["cpu_ms"] < best["cpu_ms"]:
            best = row
    best["width"] = width
    return best


def print_table(rows):
    header = ["levels", "width", "nodes", "arcs", "period",
              "cpu_ms", "gpu_ms", "speedup", "mode"]
    widths = {h: len(h) for h in header}
    disp = []
    for r in rows:
        d = {
            "levels": str(r["levels"]),
            "width": str(r["width"]),
            "nodes": str(r["nodes"]),
            "arcs": str(r["arcs"]),
            "period": "%.2f" % r["period"],
            "cpu_ms": "%.3f" % r["cpu_ms"],
            "gpu_ms": "-" if r["gpu_ms"] is None else "%.3f" % r["gpu_ms"],
            "speedup": "-" if r["speedup"] is None else "%.2fx" % r["speedup"],
            "mode": r["mode"],
        }
        disp.append(d)
        for h in header:
            widths[h] = max(widths[h], len(d[h]))

    def fmt(cells):
        return "  ".join(cells[h].rjust(widths[h]) for h in header)

    print(fmt({h: h for h in header}))
    print("  ".join("-" * widths[h] for h in header))
    for d in disp:
        print(fmt(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=os.path.join(REPO_ROOT, "build"))
    ap.add_argument("--no-build", action="store_true", help="reuse existing build")
    ap.add_argument("--reps", type=int, default=3, help="min-of-N timing (default 3)")
    ap.add_argument("--csv", default=os.path.join(REPO_ROOT, "bench", "results.csv"))
    ap.add_argument("--sizes", default=None,
                    help="comma list of LxW, e.g. 32x128,64x256 (default sweep)")
    args = ap.parse_args()

    if args.sizes:
        sizes = []
        for tok in args.sizes.split(","):
            l, w = tok.lower().split("x")
            sizes.append((int(l), int(w)))
    else:
        sizes = DEFAULT_SIZES

    if args.no_build:
        exe = os.path.join(args.build_dir, "sta")
        if not os.path.exists(exe):
            sys.exit("error: --no-build but no binary at %s (run without --no-build)" % exe)
    else:
        exe = configure_and_build(args.build_dir)

    rows = []
    any_gpu = False
    for levels, width in sizes:
        row = run_case(exe, levels, width, args.reps)
        any_gpu = any_gpu or (row["mode"] == "gpu")
        rows.append(row)

    fields = ["levels", "width", "nodes", "arcs", "period",
              "cpu_ms", "gpu_ms", "speedup", "mode"]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in fields})

    print()
    print("STA benchmark  (measured, min-of-%d)" % args.reps)
    if any_gpu:
        print("device: CUDA GPU present — GPU times/speedups are MEASURED on-device.")
    else:
        print("device: NO GPU — staGpu fell back to the CPU reference.")
        print("        gpu_ms/speedup left blank on purpose (nothing measured to report).")
    print()
    print_table(rows)
    print()
    print("wrote %s" % args.csv)


if __name__ == "__main__":
    main()
