"""Build the functional suites, one executable each; cache binaries, never test results."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
# Each tests/test_<name>.mojo is a suite with its own executable, so a
# focused suite builds without the others.
SUITES = sorted(path for path in (ROOT / "tests").glob("test_*.mojo") if not path.name.startswith("._"))
FLAGS = ("--fp-mode=contract=off", "--Werror")
# 12 GB: the project's build budget. The -O3 suite peaks near 4 GB; at -O0 it
# passed 9 GB, since unoptimized code hands the code generator far more IR.
COMPILER_MEMORY_BYTES = 12 * 1000**3


def compiler_command(command):
    """Bound compiler memory, including children; never silently run unbounded."""
    systemd = shutil.which("systemd-run")
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    controllers = Path("/sys/fs/cgroup/cgroup.controllers")
    if (systemd and runtime and (Path(runtime) / "bus").exists()
            and controllers.exists() and "memory" in controllers.read_text().split()):
        # A scope inherits the Pixi environment and process group, so the existing
        # timeout/interrupt cleanup still reaches the compiler and its children.
        return [systemd, "--user", "--scope", "--quiet",
                f"--property=MemoryMax={COMPILER_MEMORY_BYTES}",
                "--property=MemorySwapMax=0", "--property=OOMPolicy=kill", *command]
    if sys.platform == "darwin":
        # macOS has no cgroups and does not enforce an address-space limit, so
        # say so rather than run unbounded silently.
        print("Compiler guard: none on macOS; the -O3 suites peak near 4 GB.", flush=True)
        return list(command)
    prlimit = shutil.which("prlimit")
    if prlimit:
        # On Linux without a user systemd manager, bound each child's entire
        # address space. Do not apply this to ASAN executables' virtual mappings.
        print("Compiler guard: 12 GB address-space limit (systemd unavailable).", flush=True)
        return [prlimit, f"--as={COMPILER_MEMORY_BYTES}:{COMPILER_MEMORY_BYTES}",
                "--core=0:0", "--", *command]
    raise RuntimeError("Compiler memory protection is unavailable; install systemd or prlimit before building tests.")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(compiler, flags, source):
    files = [source, *sorted((ROOT / "src").rglob("*.mojo")), compiler,
             compiler.parent.parent / "lib/mojo/std.mojoc"]
    return dict(flags=flags, inputs={str(path): digest(path) for path in files if not path.name.startswith("._")})


def run(command, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Test budget exhausted; use --timeout SECONDS for a cold build.")
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", start_new_session=True) as process:
        try:
            output = process.communicate(timeout=remaining)[0]
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output = process.communicate()[0]
            if output:
                print(output, end="", flush=True)
            if time.monotonic() >= deadline:
                raise TimeoutError("Test budget exhausted; use --timeout SECONDS for a cold build.") from None
            raise
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if process.returncode:
        raise RuntimeError(f"{Path(command[0]).name} exited with status {process.returncode}.")
    return output


def scenarios(source):
    return re.findall(r"(?m)^def (test_\w+)\(", source.read_text())


def run_suite(source, cases, compiler, flags, args, deadline):
    """Build one suite unless its cached executable matches, run it, and return
    whether the build was cached."""
    directory = ROOT / "build/tests"
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / (source.stem + ("_asan" if args.asan else ""))
    stamp = binary.with_suffix(".json")
    temporary = binary.with_suffix(f".{os.getpid()}.tmp")
    try:
        inputs = identity(compiler, flags, source)
        try:
            saved = json.loads(stamp.read_text())
            cached = not args.rebuild and saved["inputs"] == inputs and saved["binary_sha256"] == digest(binary)
        except (OSError, ValueError, KeyError, TypeError):
            cached = False
        if not cached:
            cap = "no memory cap on macOS" if sys.platform == "darwin" else "12 GB memory cap"
            print(f"Building {source.name} (one compiler thread, {cap})...", flush=True)
            try:
                run(compiler_command([str(compiler), "build", *flags, str(source), "-o", str(temporary)]), deadline)
            except RuntimeError as error:
                limit = "" if sys.platform == "darwin" else " Compilation is capped at 12 GB."
                raise RuntimeError(f"{error}{limit} Inspect the compiler diagnostics before retrying.") from error
            if inputs != identity(compiler, flags, source):
                raise RuntimeError("Sources changed during compilation; rerun the tests.")
            temporary.replace(binary)
            stamp_temporary = stamp.with_suffix(f".{os.getpid()}.tmp")
            stamp_temporary.write_text(json.dumps(dict(inputs=inputs, binary_sha256=digest(binary))) + "\n")
            stamp_temporary.replace(stamp)
        output = run([str(binary)], deadline)
        summary = re.search(r"(\d+) tests run:\s*(\d+) passed\s*,\s*(\d+) failed\s*,\s*(\d+) skipped", output)
        if not summary or tuple(map(int, summary.groups())) != (len(cases), len(cases), 0, 0):
            raise RuntimeError(f"Incomplete or failing output from {source.name}.")
        if inputs != identity(compiler, flags, source):
            raise RuntimeError("Sources changed during testing; rerun the tests.")
        return cached
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=180, help="Total build/run budget in seconds (default: 180)")
    parser.add_argument("--asan", action="store_true", help="Run the suites under AddressSanitizer")
    parser.add_argument("--rebuild", action="store_true", help="Ignore the cached executables")
    parser.add_argument("--list", action="store_true", help="List the scenarios without compiling")
    parser.add_argument("--suite", choices=[path.stem.removeprefix("test_") for path in SUITES],
                        help="Run one suite, such as apn for tests/test_apn.mojo (default: all)")
    args = parser.parse_args(argv)
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be finite and positive")
    sources = [path for path in SUITES if args.suite in (None, path.stem.removeprefix("test_"))]
    suites = [(source, scenarios(source)) for source in sources]
    for source, cases in suites:
        if not cases or len(cases) != len(set(cases)):
            parser.error(f"{source.name} must have unique, named test scenarios")
    if args.list:
        for source, cases in suites:
            print("\n".join(f"{source.stem}::{case}" for case in cases))
        return 0
    executable = shutil.which("mojo")
    if not executable:
        parser.error("Mojo is unavailable; use pixi run --locked test")
    compiler = Path(executable).resolve()
    # -O3 builds faster than -O0 here and in under half the memory. Mojo 1.1.0's
    # TestSuite report trips ASAN on a zero-sized Optional at -O0; ASAN uses -O1.
    flags = ["-j1", "-O1" if args.asan else "-O3", *FLAGS,
             *(["--sanitize=address"] if args.asan else []), "-I", str(ROOT / "src")]
    started = time.monotonic()
    deadline = started + args.timeout
    try:
        for source, cases in suites:
            cached = run_suite(source, cases, compiler, flags, args, deadline)
            print(f"Passed {len(cases)} scenarios of {source.name} in {time.monotonic() - started:.2f}s"
                  f" ({'cached build' if cached else 'fresh build'}).")
        return 0
    except (OSError, RuntimeError, TimeoutError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Testing interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
