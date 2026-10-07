# Benchmarks

Use the benchmark report to compare APN Mojo with GMP, MPFR and MPC through
Rug, and FLINT and Arb through python-flint. It checks every case before
timing it. [Benchmark results](../architecture/benchmark-results.md) summarizes
the published run and links to the full report. Record measurements that
explain a design choice in the relevant architecture chapter.

Run the commands below from the project directory containing `pixi.toml`.

## Running the report

Install the comparison environment for python-flint and a Rust toolchain
(`cargo`) for the Rug worker:

```sh
pixi install --locked -e comparison
pixi run --locked report                       # the full catalog, about 5 minutes
pixi run --locked report --quick --area float  # 2 ms samples: for development
pixi run --locked report --check-only          # correctness only, seconds
pixi run --locked report --case float.add.53 --baseline HEAD
pixi run --locked report --publish             # update the summary and full report download
```

Select cases with `--area`, `--case`, `--op`, or `--bits`, and inspect them
with `--list`. Results go to `build/report/report.md` and `report.json`, which
includes every sample. A disagreement with a reference exits with status 1.
Investigate both results before deciding which library is at fault.

`--publish` writes a compact area summary to the site and preserves every
case in `docs/content/downloads/benchmarks/report.txt`, downloadable as
Markdown. Both files come from the same run; do not edit the published
numbers by hand. The local `report.json` retains the individual samples.

The apn_mojo worker builds once per set of sources, at `-O3`, with one compiler
thread under the suite's memory cap (about 85 s), and is cached under
`build/report-cache/`.

## What is compared

`tests/benchmarks/catalog.py` generates every case from one seed: about 480
cases in twelve areas, from 64-bit Integer addition to 4096-bit Ball
functions, text conversion and batches of 1000. Operands are exact text that
every worker reads alike. A Float operand is already exact at its precision, so
no backend rounds its inputs.

| Area | Reference |
|---|---|
| Integer and Rational arithmetic, number theory | GMP (Rug) and FLINT (python-flint) |
| Float arithmetic and functions | MPFR (Rug) |
| Complex arithmetic and functions | MPC (Rug) |
| Ball and ComplexBall arithmetic and functions | Arb (python-flint) |
| Text conversion | GMP and MPFR; FLINT for Integers |
| Batches | Loops over MPFR and GMP values (Rug) |

Each backend computes every case once before any timing:

- Integer, Rational, Float and Complex results must be equal. Both libraries
  round Float and Complex results correctly, so equality is exact.
- Balls must overlap, since both enclose the true value. The report also gives
  the ratio of the radii.
- Printed decimals must have the same value.

Some operations differ in what they compute:

- python-flint has no exact division, so its `div_exact` is floor division.
- GMP's primality test with 24 repetitions and FLINT's `is_probable_prime` are
  both Baillie–PSW tests, like apn_mojo's.
- In the batch area, a `_loop` case times apn_mojo's plain `List` loop of the
  scalar function. It measures the batch layer's own cost.
- The batch table also times apn_mojo on every allowed CPU. All other numbers
  use one CPU.

## How it times

Each backend runs as a persistent worker process:

- `apn_worker.mojo`;
- `rug/`, in Rust;
- `flint_worker.py`.

The driver pins all of them to one CPU (CPU 2 by default; `--cpu`), and only
one works at a time. Each case is prepared before timing; the timed call
creates its result and destroys it. Every worker follows the timeit protocol:

1. The number of calls steps 1, 2, 5, 10, 20, ... until one sample of that
   many calls takes at least the target, 10 ms by default (`--target-ms`).
2. Then seven samples (`--repeats`), each one timer pair around the loop,
   without checks.
3. The report gives the median per call and half the interquartile range.

The backends take turns case by case, in a rotating order, so slow drift in
clock speed affects all of them alike. A sentinel case is timed every 25 cases.
The spread of its medians is the run's drift, and absolute times are only as
steady as that.

python-flint is timed through the interpreter. The report times an empty call
in the same loop (`callable(a)`, or `operator.is_(a, b)`) and subtracts it from
every python-flint time. A † marks a call that takes less than four times the
overhead, since its net time is uncertain.

Stop other heavy work before measuring. The driver warns if the pinned CPU
was busy before the run. Concurrent Mojo compilation has made entire runs
several times slower.

## Before and after a change

`--baseline <commit>` builds the same worker against that commit's sources in
a temporary detached worktree and adds it as one more backend. The column
"vs baseline" is the working tree's median over the commit's, measured in the
same rotation. Select the changed functions with `--case`, `--op` or `--area`;
before a commit, that measurement is the only gate.

## Differential checks

A change that should not alter results is checked against a reference commit:

```sh
pixi run --locked python3 tests/benchmarks/differential.py --reference HEAD
pixi run --locked python3 tests/benchmarks/differential.py --only rational
```

The script builds `tests/benchmarks/differential.mojo` against a temporary
detached worktree of the reference and against the working tree, runs both, and
compares their output line by line, exiting with status 1 when a line differs.
The reference executable is kept under `build/differential/` while the
harness is unchanged. Each build is one compilation, about 20 s.

The roughly 17,800 lines cover Integer arithmetic, comparison, bitwise
operations, shifts, in-place updates, division, gcd, roots and factorials at
every seventh length up to 3300 bits and at selected lengths up to 20,000;
Rational arithmetic over 17 numerator and 13 denominator sizes; Float square
roots and arithmetic at 21 precisions in all five rounding modes, with exact,
same-precision and wider operands; and Complex square roots and arithmetic in
13 format pairs and 7 rounding-mode pairs; and Ball arithmetic, `sqrt` and six
functions on balls of eight precisions with five radii each, in three result
precisions. Values print as their residue modulo 1000000007, bit length and
sign, Float results with their status flags, and balls with the radius's
mantissa and exponent; `--only ball` runs that section alone. Each Ball line
starts with its operation, so a change meant to alter one radius shows on that
operation's lines alone.

## Attributing time

When hardware performance counters (`perf`) are restricted in your development
environment, compile with `--debug-level=line-tables` and profile instruction
counts using Callgrind:

```sh
valgrind --tool=callgrind --callgrind-out-file=.cache/out ./harness
callgrind_annotate --inclusive=yes .cache/out
```

The report's apn_mojo worker accepts `calls i n`, which makes exactly `n` timed
calls of case `i` and nothing else, so counts repeat between builds. The
report prints the worker's path and writes its catalog, `cases.tsv`, whose
line `i` (from 0) is case `i`:

```sh
pixi run --locked report --case integer.gcd.1024 --check-only --line-tables
printf 'calls 0 20\nquit\n' | valgrind --tool=callgrind <worker> build/report/cases.tsv
```

Instruction counts miss memory stalls and cache effects. Returning a large
struct across a non-inlined call, for example, can add substantial latency
without many instructions. Confirm a suspected bottleneck with a targeted
change and a wall-clock measurement.
