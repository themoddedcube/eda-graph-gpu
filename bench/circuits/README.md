# ISCAS-85 combinational benchmark circuits

Standard, public-domain academic gate-level benchmarks (combinational, `.bench`
format), used here as REAL netlists to validate the STA primitive on non-synthetic
topology. Fetched from the ISCAS-85 mirror at
`https://pld.ttu.ee/~maksim/benchmarks/iscas85/bench/`.

`.bench` format: `INPUT(net)`, `OUTPUT(net)`, and `out = GATE(in, in, ...)` where
GATE ∈ {AND, OR, NAND, NOR, NOT, BUFF, XOR}. Parsed by `readBench()`
(`src/circuit.cpp`), which levelizes the gate DAG into a `TimingGraph`.

| circuit | gates | note |
|---|---:|---|
| c17 | 6 | smallest; hand-verifiable (period 6) |
| c432 … c7552 | 160 … 3512 | the classic ISCAS-85 set |

Delays here use a nominal unit-ish model (BUFF/NOT=1, XOR=3, else 2) — real STA uses a
cell timing library; this is a documented stand-in for topology/correctness testing.
