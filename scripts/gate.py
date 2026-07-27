"""Run every quality gate, report each one, and fail on the first that is not clean.

One definition of "green", called by three callers: a developer running it by
hand, the pre-commit hook, and CI. Keeping it here rather than duplicating the
command list in a workflow file is what stops CI and the hook from drifting into
checking different things.

    uv run python scripts/gate.py           # everything
    uv run python scripts/gate.py --fast    # skip the slow checks, for the hook

Exit code is 0 only when every gate selected has passed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    name: str
    command: tuple[str, ...]
    # False for gates too slow to sit in a pre-commit hook.
    fast: bool
    why: str


GATES: tuple[Gate, ...] = (
    Gate(
        name="ruff",
        command=("ruff", "check", "src", "tests", "conftest.py", "scripts"),
        fast=True,
        why="lint, including complexity and the bandit security rules",
    ),
    Gate(
        name="format",
        command=("ruff", "format", "--check", "src", "tests", "conftest.py", "scripts"),
        fast=True,
        why="formatting is decided by the tool, not by review",
    ),
    Gate(
        name="mypy",
        command=("mypy",),
        fast=True,
        why="types",
    ),
    Gate(
        name="imports",
        command=("lint-imports",),
        fast=True,
        why="the pure core must not reach for transport, which is what keeps the suite offline",
    ),
    Gate(
        name="deptry",
        command=("deptry", "src", "tests"),
        fast=True,
        why="declared dependencies match imported ones",
    ),
    Gate(
        name="tests",
        command=("pytest", "-q", "--cov", "--cov-report=term-missing"),
        fast=False,
        why="the suite, with the coverage floor",
    ),
)


def run(gate: Gate) -> tuple[bool, float]:
    started = time.monotonic()
    # Fixed tuple, no shell. Every command is a literal defined above, so there
    # is nothing for a shell to interpolate and nothing to quote.
    completed = subprocess.run(gate.command, check=False)
    return completed.returncode == 0, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only the gates quick enough for a pre-commit hook.",
    )
    args = parser.parse_args()

    selected = [gate for gate in GATES if gate.fast or not args.fast]
    failures: list[str] = []

    for gate in selected:
        print(f"\n=== {gate.name}: {gate.why} ===", flush=True)
        try:
            passed, seconds = run(gate)
        except FileNotFoundError:
            # A missing tool is an unknown result, never a pass. Reporting it as
            # clean is how a gate goes quietly blind.
            print(f"FAIL {gate.name}: '{gate.command[0]}' not found. Run: uv sync")
            failures.append(gate.name)
            continue
        print(f"{'PASS' if passed else 'FAIL'} {gate.name} ({seconds:.1f}s)")
        if not passed:
            failures.append(gate.name)

    print("\n" + "=" * 60)
    if failures:
        print(f"GATE FAILED: {', '.join(failures)}")
        return 1
    print(f"GATE PASSED: {len(selected)} checks clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
