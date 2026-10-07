"""Check that a change leaves every result unchanged: build
tests/benchmarks/differential.mojo against a reference commit and against the
working tree, run both, and compare their output line by line.

    pixi run --locked python3 tests/benchmarks/differential.py [--reference HEAD] [--only SECTION]

The reference build uses a temporary detached worktree of the commit, removed
after the build; its executable is kept under build/differential/<commit>/ and
reused while differential.mojo is unchanged. Both builds compile this tree's
differential.mojo, so they run the same inputs. Each build is one compilation: run nothing else that compiles alongside.
Exits 1 when any line differs.
"""

import argparse
from pathlib import Path
import subprocess
import sys

from reference_build import build, reference_build

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / "build/differential"
SOURCE = HERE / "differential.mojo"
FLAGS = ["-j1", "-O3", "--fp-mode=contract=off"]


def run(executable, only):
    command = [str(executable)] + ([only] if only else [])
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", default="HEAD", help="Commit to compare against (default HEAD)")
    parser.add_argument("--only", choices=("integer", "rational", "float", "complex", "ball"), help="Run one section")
    parser.add_argument("--rebuild-reference", action="store_true", help="Rebuild a cached reference executable")
    parser.add_argument("--no-build", action="store_true", help="Reuse the existing candidate executable")
    args = parser.parse_args()
    before, sha = reference_build(args.reference, SOURCE, FLAGS, OUTPUT, args.rebuild_reference)
    after = OUTPUT / "candidate/differential"
    if not args.no_build:
        build(SOURCE, FLAGS, ROOT, after)
    expected, observed = run(before, args.only), run(after, args.only)
    differences = [(i, a, b) for i, (a, b) in enumerate(zip(expected, observed)) if a != b]
    if len(expected) != len(observed):
        differences.append((min(len(expected), len(observed)), f"{len(expected)} lines", f"{len(observed)} lines"))
    print(f"{len(observed)} lines against {sha[:12]}; {len(differences)} differ")
    for index, a, b in differences[:10]:
        print(f"line {index + 1}:\n  reference: {a[:200]}\n  candidate: {b[:200]}")
    return int(bool(differences))


if __name__ == "__main__":
    sys.exit(main())
