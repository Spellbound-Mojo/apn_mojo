"""The consolidated report: apn_mojo against GMP, MPFR and MPC (through Rug)
and FLINT and Arb (through python-flint), on the cases of catalog.py.

Each backend is a persistent worker process: apn_worker.mojo, rug/ and
flint_worker.py. All of them are pinned to one CPU, and only one is active at
a time.

The run has two phases:

- Correctness. Every backend computes every case once. Results must be
  equal, or, for balls, overlap.
- Timing, in the timeit style and without checks. The number of calls steps
  1, 2, 5, 10, ... until one sample takes the target time; then `--repeats`
  samples. The report gives each median per call, absolute, with its
  interquartile range. The backends take turns case by case, in a rotating
  order. A sentinel case is timed every 25 cases to measure drift.

python-flint's times are also given net of an empty call, which is the loop
and the interpreter's call overhead.

The output is a JSON record and a Markdown report under `--output`.
`--publish` writes a compact summary and a downloadable full report to the
documentation site.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import queue
import runpy
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import catalog

sys.set_int_max_str_digits(0)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
compiler_command = runpy.run_path(str(HERE.parent / "run.py"))["compiler_command"]
FLAGS = ["-j1", "-O3", "--fp-mode=contract=off", "--Werror"]
PAGE = ROOT / "docs/content/architecture/benchmark-results.md"
DETAILS = ROOT / "docs/content/downloads/benchmarks/report.txt"
SENTINEL = "integer.multiply.1024"
NAMES = dict(apn_mojo="apn_mojo", rug="Rug", flint="FLINT/Arb", apn_baseline="baseline")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*arguments):
    return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def build_apn_worker(output, flags=FLAGS):
    """The Mojo worker, built once per set of sources, flags and compiler."""
    mojo = shutil.which("mojo")
    if not mojo:
        raise RuntimeError("Mojo not found; run through `pixi run --locked report`")
    compiler = Path(mojo).resolve()
    sources = [*sorted((ROOT / "src").rglob("*.mojo")), HERE / "apn_worker.mojo", compiler,
               compiler.parent.parent / "lib/mojo/std.mojoc"]
    key = hashlib.sha256(json.dumps(dict(files={str(p): digest(p) for p in sources}, flags=flags),
                                    sort_keys=True).encode()).hexdigest()
    executable = ROOT / "build/report-cache" / key / "apn-worker"
    if executable.exists():
        return executable
    executable.parent.mkdir(parents=True, exist_ok=True)
    print("Building the apn_mojo worker (one compiler thread, memory-capped)...", flush=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=executable.parent) as temporary:
        candidate = Path(temporary) / "apn-worker"
        with open(output / "apn-build.log", "w") as log:
            built = subprocess.run(compiler_command([mojo, "build", *flags, "-I", str(ROOT / "src"),
                                                     str(HERE / "apn_worker.mojo"), "-o", str(candidate)]),
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if built.returncode:
            raise RuntimeError(f"The apn_mojo worker did not build; see {output / 'apn-build.log'}")
        candidate.replace(executable)
    print(f"Built in {time.monotonic() - started:.0f} s", flush=True)
    return executable


def build_rug_worker(output):
    cargo = shutil.which("cargo")
    if not cargo:
        raise RuntimeError("Cargo not found; the Rug worker needs a Rust toolchain")
    with open(output / "rug-build.log", "w") as log:
        built = subprocess.run([cargo, "build", "--release", "--locked", "--manifest-path",
                                str(HERE / "rug/Cargo.toml"), "--target-dir", str(ROOT / "build/rug-worker")],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if built.returncode:
        raise RuntimeError(f"The Rug worker did not build; see {output / 'rug-build.log'}")
    return ROOT / "build/rug-worker/release/apn-reference-worker"


class Worker:
    """One backend's process: a command line in, one JSON line out."""

    def __init__(self, name, command, output, timeout, cpus=None):
        self.name, self.timeout = name, timeout
        self.log = open(output / f"{name}.stderr.log", "w")
        self.process = subprocess.Popen(command, cwd=ROOT, text=True, bufsize=1, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log,
                                        preexec_fn=(lambda: os.sched_setaffinity(0, cpus)) if cpus else None)
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def ask(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            self.process.kill()
            raise TimeoutError(f"{self.name} did not answer `{command}` within {self.timeout} s") from None
        if line is None:
            raise RuntimeError(f"{self.name} exited at `{command}`; see {self.log.name}")
        return json.loads(line)

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write("quit\n")
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
        self.log.close()


def number(record):
    """A float record, from apn_mojo's JSON or Rug's, as (class, negative, value)."""
    kind = record["class"]
    negative = record.get("negative", record.get("sign") == "-")
    if kind != "finite":
        return ({"infinity": "inf", "infinite": "inf"}.get(kind, kind), negative if kind != "nan" else None, None)
    # apn_mojo writes an unsigned significand and a sign; Rug a signed significand.
    value = int(record["significand"]) * Fraction(2) ** (int(record["exponent"]) - int(record.get("precision", 0)))
    if record.get("sign") == "-":
        value = -value
    return ("finite", value < 0, value)


def canonical(value):
    """A comparable form of any worker's result."""
    if "list" in value:
        return ("list", tuple(canonical(v) for v in value["list"]))
    if "text" in value:
        return ("text", value["text"])
    if "ball" in value:
        return ("ball", Fraction(value["ball"]["mid"]), Fraction(value["ball"]["rad"]))
    if value.get("family") == "ball":
        if value["kind"] != "finite":
            return ("ball", value["kind"])
        mid, rad = number(value["midpoint"]), number(value["radius"])
        return ("ball", mid[2] or Fraction(0), rad[2] or Fraction(0))
    if "real" in value:
        return ("pair", canonical(value["real"]), canonical(value["imag"]))
    if "integer" in value or value.get("family") == "integer":
        return ("integer", int(value.get("integer", value.get("value"))))
    if "numerator" in value:
        return ("rational", Fraction(int(value["numerator"]), int(value["denominator"])))
    return ("float",) + number(value)


def overlap(a, b):
    """Whether two canonical balls or ball pairs overlap, and the ratio of their radii."""
    if a[0] == "pair":
        (re, r1), (im, r2) = overlap(a[1], b[1]), overlap(a[2], b[2])
        return re and im, (r1 * r2) ** 0.5 if r1 and r2 else None
    if len(a) != 3 or len(b) != 3:
        return False, None
    meets = abs(a[1] - b[1]) <= a[2] + b[2]
    return meets, float(a[2] / b[2]) if b[2] else None


def compare(case, apn, reference):
    """(agrees, radius ratio or None) for one reference's result."""
    a, b = canonical(apn), canonical(reference)
    if case["check"] == "ball":
        return overlap(a, b)
    if case["check"] == "decimal":
        return Decimal(a[1]) == Decimal(b[1]), None
    return a == b, None


def statistics_of(answer):
    """Median and interquartile range per call, in ns."""
    per_call = [s / answer["number"] for s in answer["samples"]]
    q1, median, q3 = statistics.quantiles(per_call, n=4, method="inclusive")
    return dict(median=statistics.median(per_call), iqr=q3 - q1, number=answer["number"], per_call=per_call)


def cpu_busy(cpu, seconds=1.0):
    """The fraction of `cpu`'s time spent busy over `seconds`, before the run."""
    def read():
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith(f"cpu{cpu} "):
                values = [int(v) for v in line.split()[1:]]
                return sum(values), values[3] + values[4]
    try:
        total0, idle0 = read()
        time.sleep(seconds)
        total1, idle1 = read()
        return 1 - (idle1 - idle0) / max(1, total1 - total0)
    except (OSError, TypeError):
        return None


def environment(cpu, workers):
    def first(path, default="unknown"):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return default
    model = next((line.split(":", 1)[1].strip() for line in first("/proc/cpuinfo", "").splitlines()
                  if line.startswith("model name")), platform.processor())
    mojo = subprocess.run([shutil.which("mojo") or "mojo", "--version"], capture_output=True, text=True).stdout.strip()
    return dict(
        date=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        commit=git("rev-parse", "HEAD"), modified=bool(git("status", "--porcelain", "--", "src")),
        version=tomllib.loads((ROOT / "pixi.toml").read_text())["workspace"]["version"],
        cpu=model, pinned_cpu=cpu, governor=first(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"),
        boost=first("/sys/devices/system/cpu/intel_pstate/no_turbo", None) == "0" if Path(
            "/sys/devices/system/cpu/intel_pstate/no_turbo").exists() else None,
        kernel=platform.release(), mojo=mojo,
        versions={name: w.ask("version") for name, w in workers.items() if name in ("rug", "flint")},
    )


def write_catalog(path, cases):
    with open(path, "w") as stream:
        for c in cases:
            stream.write("\t".join([c["id"], c["family"], c["op"], str(c["bits"]), str(c["param"]), *c["operands"]]) + "\n")


def run(args, cases, output):
    every = catalog.cases()
    if not any(c["id"] == SENTINEL for c in cases):
        cases = cases + [c for c in every if c["id"] == SENTINEL]
        sentinel_only = {SENTINEL}
    else:
        sentinel_only = set()
    index = {c["id"]: i for i, c in enumerate(cases)}
    write_catalog(output / "cases.tsv", cases)

    flags = FLAGS + (["--debug-level=line-tables"] if args.line_tables else [])
    executables = dict(apn_mojo=[str(build_apn_worker(output, flags))], rug=[str(build_rug_worker(output))],
                       flint=[str(ROOT / ".pixi/envs/comparison/bin/python"), str(HERE / "flint_worker.py")])
    if not Path(executables["flint"][0]).exists():
        raise RuntimeError("python-flint needs the comparison environment: pixi install -e comparison")
    if args.baseline:
        from reference_build import reference_build
        baseline, sha = reference_build(args.baseline, HERE / "apn_worker.mojo", FLAGS, ROOT / "build/report-baseline")
        executables["apn_baseline"] = [str(baseline)]
        args.baseline_commit = sha
    # Batches are also timed on every CPU the run may use; everything else on one.
    allowed = os.sched_getaffinity(0)
    parallel = any(c["area"] == "batch" for c in cases) and len(allowed) > 1
    # Pinned after the builds, so that compilers keep every CPU.
    os.sched_setaffinity(0, {args.cpu})
    busy = cpu_busy(args.cpu)
    if busy is not None and busy > 0.05:
        print(f"Warning: CPU {args.cpu} was {busy:.0%} busy before the run; timings will be slow and noisy.", flush=True)
    workers = {}
    try:
        for name, command in executables.items():
            workers[name] = Worker(name, [*command, str(output / "cases.tsv")], output, args.call_timeout)
        if parallel:
            workers["apn_parallel"] = Worker("apn_parallel", [*executables["apn_mojo"], str(output / "cases.tsv")],
                                             output, args.call_timeout, cpus=allowed)
        report = dict(environment=environment(args.cpu, workers), cpu_busy_before=busy, target_ns=args.target_ns,
                      repeats=args.repeats, baseline=getattr(args, "baseline_commit", None),
                      apn_worker=executables["apn_mojo"][0], parallel_cpus=len(allowed) if parallel else None, cases=[])
        print(f"apn_mojo worker: {executables['apn_mojo'][0]}; catalog: {output / 'cases.tsv'}", flush=True)

        print(f"Checking {len(cases) - len(sentinel_only)} cases once per backend...", flush=True)
        rows = []
        for case in cases:
            if case["id"] in sentinel_only:
                continue
            i = index[case["id"]]
            backends = ["apn_mojo", *case["references"], *(["apn_baseline"] if args.baseline else [])]
            answers = {b: workers[b].ask(f"check {i}") for b in backends}
            row = dict(case={k: case[k] for k in ("id", "area", "family", "op", "bits", "param", "references", "check")},
                       errors={b: a["error"] for b, a in answers.items() if "error" in a}, agrees={}, timing={})
            if "flint" in answers and "arity" in answers["flint"]:
                row["arity"] = answers["flint"]["arity"]
            if "apn_mojo" not in row["errors"]:
                for b in backends[1:]:
                    if b not in row["errors"]:
                        agrees, ratio = compare(case, answers["apn_mojo"]["result"], answers[b]["result"])
                        row["agrees"][b] = agrees
                        if ratio is not None and b != "apn_baseline":
                            row["radius_ratio"] = ratio
                        if not agrees:
                            row.setdefault("results", {})[b] = answers[b]["result"]
                            row["results"]["apn_mojo"] = answers["apn_mojo"]["result"]
            rows.append(row)
        failed = [r["case"]["id"] for r in rows if r["errors"] or not all(r["agrees"].values())]
        print(f"{len(rows) - len(failed)} of {len(rows)} cases agree" + (f"; disagreements: {', '.join(failed)}" if failed else ""),
              flush=True)
        if args.check_only:
            report["cases"] = rows
            return report

        if any("flint" in r["case"]["references"] for r in rows):
            report["flint_empty_call"] = {str(n): statistics_of(workers["flint"].ask(
                f"empty {n} {args.target_ns} {args.repeats}")) for n in (1, 2)}

        print(f"Timing: {args.target_ns / 1e6:g} ms per sample, {args.repeats} samples...", flush=True)
        report["sentinel"] = []
        started = time.monotonic()

        def sentinel(n):
            sample = statistics_of(workers["rug"].ask(f"time {index[SENTINEL]} {args.target_ns} {args.repeats}"))
            report["sentinel"].append(dict(after_cases=n, seconds=time.monotonic() - started, median=sample["median"]))
            return sample["median"]

        def time_row(n, row):
            backends = [b for b in ["apn_mojo", *row["case"]["references"], *(["apn_baseline"] if args.baseline else []),
                                    *(["apn_parallel"] if parallel and row["case"]["area"] == "batch" else [])]
                        if b not in row["errors"] and (b != "apn_parallel" or "apn_mojo" not in row["errors"])]
            shift = n % len(backends)
            for b in backends[shift:] + backends[:shift]:
                row["timing"][b] = statistics_of(workers[b].ask(
                    f"time {index[row['case']['id']]} {args.target_ns} {args.repeats}"))

        marks = []
        for n, row in enumerate(rows):
            if n % 25 == 0:
                marks.append((n, sentinel(n)))
                if n:
                    rate = (time.monotonic() - started) / n
                    print(f"  {n}/{len(rows)} cases, about {rate * (len(rows) - n) / 60:.1f} min left", flush=True)
            time_row(n, row)
        marks.append((len(rows), sentinel(len(rows))))
        # A checkpoint well above the typical sentinel means other work shared the
        # CPU around it: the cases on both sides of it are timed again.
        typical = statistics.median(m for _, m in marks)
        disturbed = [i for i, (_, m) in enumerate(marks) if m > 1.15 * typical]
        blocks = sorted({b for i in disturbed for b in (i - 1, i) if 0 <= b < len(marks) - 1})
        for b in blocks:
            print(f"  timing cases {marks[b][0]}-{marks[b + 1][0] - 1} again: the sentinel ran slow next to them", flush=True)
            for n in range(marks[b][0], marks[b + 1][0]):
                time_row(n, rows[n])
            sentinel(marks[b + 1][0])
        medians = [s["median"] for s in report["sentinel"]]
        steady = [m for m in medians if m <= 1.15 * typical]
        report["drift"] = max(steady) / min(steady) - 1
        report["disturbed"] = dict(checkpoints=len(disturbed), retimed_cases=sum(marks[b + 1][0] - marks[b][0] for b in blocks),
                                   slow_after_retiming=sum(m > 1.15 * typical for m in medians[len(marks):]))
        report["timing_seconds"] = time.monotonic() - started
        report["cases"] = rows
        return report
    finally:
        for w in workers.values():
            w.close()


def duration(ns):
    """Three significant digits and a unit: 23.0 ns, 1.25 µs."""
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3), ("ns", 1)):
        if ns >= scale or unit == "ns":
            return f"{ns / scale:#.3g}".rstrip(".") + " " + unit


def net_flint(row, report):
    """python-flint's median less the empty call of the same arity; None when that leaves nothing."""
    empty = report["flint_empty_call"]["1" if row.get("arity", 2) == 1 else "2"]["median"]
    net = row["timing"]["flint"]["median"] - empty
    return net if net > 0 else None, row["timing"]["flint"]["median"] < 4 * empty


def ratios(row, report):
    """apn_mojo's median over each reference's, for cases that agree."""
    out = {}
    apn = row["timing"].get("apn_mojo")
    for b, t in row["timing"].items():
        if b in ("apn_mojo", "apn_parallel", "apn_baseline") or not apn or not row["agrees"].get(b, False):
            continue
        reference = net_flint(row, report)[0] if b == "flint" else t["median"]
        if reference:
            out[b] = apn["median"] / reference
    return out


def geometric_mean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def label(case):
    """The operation as the case ID names it: `floordiv_small`, `float_exp`."""
    return case["id"].split(".")[1]


def size(case):
    if case["op"] in ("factorial", "binomial"):
        return f"n = {case['bits']}"
    return str(case["bits"])


def minutes(seconds):
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min"


def markdown(report, *, compact=False):
    """Render the full report, or its compact documentation summary."""
    env = report["environment"]
    rows = report["cases"]
    versions = env["versions"]
    rug, flint = versions.get("rug", {}), versions.get("flint", {})
    empty = report.get("flint_empty_call", {})
    lines = [
        "# Benchmark results",
        "",
        "<!-- Generated by tests/benchmarks/report.py --publish; do not edit by hand. -->",
        "",
        *([f"> **Disturbed run.** Other work shared the CPU. After re-timing,"
           f" {report['disturbed']['slow_after_retiming']} sentinel"
           f" checkpoint{'s' if report['disturbed']['slow_after_retiming'] != 1 else ''} remained slow,"
           " so some times below are too long."
           " Before quoting them, regenerate the page on a quiet machine with `pixi run --locked report --publish`.", ""]
          if report.get("disturbed", {}).get("slow_after_retiming") else []),
        "This report compares APN Mojo with GMP, MPFR and MPC through Rug, and",
        "with FLINT and Arb through python-flint. For the workloads, measurement",
        "method, and commands, see [Benchmarks](../contributing/benchmarks.md).",
        "",
        "## How these were measured",
        "",
        f"- **Recorded:** {env['date']}. This is a snapshot of the stated build, not a measurement of the current checkout.",
        f"- **Machine:** {env['cpu']}, CPU {env['pinned_cpu']} only, governor `{env['governor']}`"
        + ("" if env["boost"] is None else f", turbo {'on' if env['boost'] else 'off'}") + f"; Linux {env['kernel']}.",
        f"- **Software:** apn_mojo {env['version']}" + (" with local changes" if env["modified"] else "")
        + f", {env['mojo']}; GMP {rug.get('gmp')}, MPFR {rug.get('mpfr')}, MPC {rug.get('mpc')} through Rug"
        f" {rug.get('rug')}; FLINT {flint.get('flint')} through python-flint {flint.get('python-flint')}"
        f" on Python {flint.get('python')}.",
        f"- **Correctness:** each case ran once on every backend before timing. Integer, Rational, Float",
        "  and Complex results must be equal; Float and Complex results are correctly rounded in",
        "  both libraries. Balls must overlap. "
        + f"{sum(1 for r in rows if not r['errors'] and all(r['agrees'].values()))} of {len(rows)} cases agree.",
        f"- **Timing:** following the timeit protocol, calls increase through 1, 2, 5, 10, ...",
        f"  until a sample takes {report['target_ns'] / 1e6:g} ms, then {report['repeats']} samples are collected.",
        "  The full report gives medians per call; ± is half the interquartile range relative to the median.",
        "  Inputs are prepared before timing, and each timed call creates and destroys its result",
        "  without checks. Backends take turns case by case.",
    ]
    if empty:
        lines += [
            f"- **python-flint overhead:** an empty call in the same loop takes {duration(empty['1']['median'])}"
            f" with one argument and {duration(empty['2']['median'])} with two. Comparisons with FLINT/Arb subtract it.",
            "  † marks a call less than four times the overhead; its net time is uncertain.",
        ]
    lines += [
        f"- **Drift:** a sentinel case, Rug's 1024-bit Integer product, was timed every 25 cases; its median"
        f" varied by {report['drift']:.1%} over the run's {minutes(report['timing_seconds'])}."
        + (f" {report['disturbed']['checkpoints']} checkpoint{'s' if report['disturbed']['checkpoints'] != 1 else ''}"
           f" ran more than 15% slow, a sign of other work on"
           f" the CPU; the {report['disturbed']['retimed_cases']} cases next to them were timed again"
           + (f". Checkpoints still slow afterwards: {report['disturbed']['slow_after_retiming']}."
              if report['disturbed']['slow_after_retiming'] else ".")
           if report["disturbed"]["checkpoints"] else ""),
        "",
        *([] if compact else [
            "The ÷ columns divide APN Mojo's time by the reference's. A ratio below 1 means APN Mojo was faster in that case.",
            "",
        ]),
        "## Summary",
        "",
        "Each ratio is a geometric mean of APN Mojo / reference times for matching cases",
        "whose results agree. **0.5 means half the time; 2 means twice the time.**",
        "Averages span the tested sizes and operations, so they are not a prediction for every call.",
        "A dash means no comparison; Text conversion has FLINT results only for Integers.",
        "",
        "| Area | Cases | APN / Rug | APN / FLINT or Arb |",
        "|---|---:|---:|---:|",
    ]
    for area, title in catalog.AREAS.items():
        group = [r for r in rows if r["case"]["area"] == area]
        if not group:
            continue
        means = [geometric_mean([ratios(r, report)[b] for r in group if b in ratios(r, report)]) for b in ("rug", "flint")]
        area_label = title if compact else f"[{title}](#{title.lower().replace(' ', '-')})"
        lines.append(f"| {area_label} | {len(group)} | "
                     + " | ".join(f"{m:.2f}" if m else "–" for m in means) + " |")
    if compact:
        lines += [
            "",
            "Rug supplies GMP for exact arithmetic, MPFR for Float, and MPC for Complex.",
            "The other column uses FLINT for exact arithmetic and Arb for balls.",
            "Ball overlap checks consistency of enclosures; it does not mean their radii are equal.",
            "",
            "## Detailed results",
            "",
            "[Download the full report](../downloads/benchmarks/report.txt){download=\"benchmark-report.md\"}",
            "for every operation and precision, absolute timings, sample variation, ball radii,",
            "and batch timings on multiple CPUs. It is a Markdown file you can search or render locally.",
            "",
            "Short python-flint calls marked † in that report are sensitive to overhead subtraction;",
            "their ratios also contribute to the averages above. Check those rows before interpreting",
            "small differences. All summary ratios use the single-CPU measurements, including batches.",
            "",
            "To measure a particular workload, follow the [benchmark guide](../contributing/benchmarks.md).",
            "Local runs retain the full report and the individual samples in `build/report/`.",
        ]
    detail_areas = () if compact else catalog.AREAS.items()
    for area, title in detail_areas:
        group = [r for r in rows if r["case"]["area"] == area]
        if not group:
            continue
        references = [b for b in ("rug", "flint") if any(b in r["case"]["references"] for r in group)]
        balls = any(r["case"]["check"] == "ball" for r in group)
        baseline = report.get("baseline") is not None
        header = ["Operation", "Bits", "apn_mojo"]
        for b in references:
            header += [NAMES[b] + (" (net)" if b == "flint" else ""), "÷ " + NAMES[b]]
        parallel = area == "batch" and report.get("parallel_cpus")
        if parallel:
            header.append(f"apn_mojo, {report['parallel_cpus']} CPUs")
        if balls:
            header.append("Radius")
        if baseline:
            header.append("vs baseline")
        lines += ["", f"## {title}", "", "| " + " | ".join(header) + " |",
                  "|---|---:|" + "---:|" * (len(header) - 2)]
        for r in group:
            case, timing = r["case"], r["timing"]

            def shown(b):
                if b in r["errors"]:
                    return "error"
                t = timing.get(b)
                return f"{duration(t['median'])} ±{t['iqr'] / 2 / t['median']:.1%}" if t else "–"
            cells = [label(case), size(case), shown("apn_mojo")]
            ratio = ratios(r, report)
            for b in references:
                if b not in case["references"]:
                    cells += ["–", "–"]
                    continue
                if b == "flint" and "flint" in timing:
                    net, uncertain = net_flint(r, report)
                    cells.append((duration(net) if net else "≈ 0") + (" †" if uncertain else ""))
                else:
                    cells.append(shown(b))
                agrees = r["agrees"].get(b)
                cells.append("**differs**" if agrees is False else f"{ratio[b]:.2f}" if b in ratio else "–")
            if parallel:
                cells.append(shown("apn_parallel"))
            if balls:
                cells.append(f"{r['radius_ratio']:.2g}" if r.get("radius_ratio") else "–")
            if baseline:
                base, apn = timing.get("apn_baseline"), timing.get("apn_mojo")
                cells.append(f"{apn['median'] / base['median']:.3f}" if base and apn else "–")
            lines.append("| " + " | ".join(cells) + " |")
        if balls:
            lines += ["", "The Radius column divides APN Mojo's radius by Arb's. For ComplexBall, it gives the geometric mean of the two component ratios."]
    problems = [r for r in rows if r["errors"] or not all(r["agrees"].values())]
    if problems:
        lines += ["", "## Disagreements", ""]
        for r in problems:
            what = ", ".join([f"{b}: {e}" for b, e in r["errors"].items()]
                             + [f"differs from {NAMES[b]}" for b, ok in r["agrees"].items() if not ok])
            lines.append(f"- `{r['case']['id']}`: {what}")
    return "\n".join(lines) + "\n"


def publish(report, full_report):
    """Keep the site's summary and downloadable details from the same run."""
    DETAILS.parent.mkdir(parents=True, exist_ok=True)
    DETAILS.write_text(full_report)
    PAGE.write_text(markdown(report, compact=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", action="append", choices=list(catalog.AREAS), help="Areas to run; repeatable")
    parser.add_argument("--case", action="append", help="Case IDs to run; repeatable")
    parser.add_argument("--op", action="append", help="Operations to run; repeatable")
    parser.add_argument("--bits", type=int, action="append", help="Sizes to run; repeatable")
    parser.add_argument("--quick", action="store_true", help="2 ms samples, 5 repeats: for development")
    parser.add_argument("--target-ms", type=float, default=10, help="Minimum time of one sample (default 10)")
    parser.add_argument("--repeats", type=int, default=7, help="Samples per case and backend (default 7)")
    parser.add_argument("--cpu", type=int, default=2, help="The CPU every worker is pinned to (default 2)")
    parser.add_argument("--baseline", help="Also time apn_mojo at this commit, built from this worker")
    parser.add_argument("--call-timeout", type=float, default=600, help="Seconds a worker may take per command")
    parser.add_argument("--output", type=Path, default=ROOT / "build/report")
    parser.add_argument("--publish", action="store_true", help="Write the site's summary and downloadable full report")
    parser.add_argument("--list", action="store_true", help="List the selected cases and exit")
    parser.add_argument("--check-only", action="store_true", help="Run the correctness phase only")
    parser.add_argument("--line-tables", action="store_true",
                        help="Build the apn_mojo worker with line tables, so callgrind names its functions")
    args = parser.parse_args(argv)
    if args.quick:
        args.target_ms, args.repeats = 2, 5
    if args.repeats < 3 or args.target_ms <= 0:
        parser.error("Use at least 3 repeats and a positive target")
    args.target_ns = int(args.target_ms * 1e6)
    cases = [c for c in catalog.cases()
             if (not args.area or c["area"] in args.area) and (not args.case or c["id"] in args.case)
             and (not args.op or c["op"] in args.op) and (not args.bits or c["bits"] in args.bits)]
    if not cases:
        parser.error("No cases match")
    if args.list:
        print("\n".join(c["id"] for c in cases))
        return 0
    if args.publish and (args.area or args.case or args.op or args.bits or args.quick or args.baseline
                         or args.check_only):
        parser.error("--publish takes the full catalog at the default protocol")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = run(args, cases, output)
    if args.check_only:
        (output / "check.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
        return int(any(r["errors"] or not all(r["agrees"].values()) for r in report["cases"]))
    (output / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    text = markdown(report)
    (output / "report.md").write_text(text)
    if args.publish:
        publish(report, text)
    disagreements = sum(1 for r in report["cases"] if r["errors"] or not all(r["agrees"].values()))
    print(f"Report: {output / 'report.md'}" + (f"; published to {PAGE.relative_to(ROOT)}" if args.publish else "")
          + f". Drift {report['drift']:.1%}; {disagreements} disagreements.")
    return int(bool(disagreements))


if __name__ == "__main__":
    raise SystemExit(main())
