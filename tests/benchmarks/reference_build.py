"""Builds of a benchmark harness against the working tree or a commit's sources,
shared by differential.py and report.py. Each build is one compilation."""

import hashlib
from pathlib import Path
import runpy
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
compiler_command = runpy.run_path(str(HERE.parent / "run.py"))["compiler_command"]


def git(*arguments):
    return subprocess.run(["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def build(source, flags, source_root, executable):
    """Compile `source` against `source_root`/src into `executable`."""
    mojo = shutil.which("mojo") or str(ROOT / ".pixi/envs/default/bin/mojo")
    executable.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    subprocess.run(compiler_command([mojo, "build", *flags, "-I", str(source_root / "src"), str(source),
                                     "-o", str(executable)]), cwd=ROOT, check=True)
    print(f"Built {executable.relative_to(ROOT)} in {time.monotonic() - started:.0f} s", flush=True)


def reference_build(commit, source, flags, cache, rebuild=False):
    """`source` from this tree, built against a commit's sources in a temporary
    detached worktree. The executable is kept under cache/<commit>/<digest of
    source>/ and reused while the harness is unchanged. Returns the executable
    and the commit's full hash."""
    sha = git("rev-parse", "--verify", commit + "^{commit}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    executable = cache / sha / digest / source.stem
    if executable.exists() and not rebuild:
        return executable, sha
    worktree = cache / ("worktree-" + sha[:12])
    if worktree.exists():
        git("worktree", "remove", "--force", str(worktree))
    git("worktree", "add", "--detach", str(worktree), sha)
    try:
        build(source, flags, worktree / git("rev-parse", "--show-prefix"), executable)
    finally:
        git("worktree", "remove", "--force", str(worktree))
    return executable, sha
