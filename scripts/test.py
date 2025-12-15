#!/usr/bin/env python3
"""
Pytest runner used by `make test*` targets.

This repo historically referenced `scripts/test.py` from the Makefile; this script
provides a stable interface for running the full suite or subsets by marker.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Optional


def _build_pytest_args(
    target: str,
) -> list[str]:
    marker: Optional[str] = None

    if target == "full":
        marker = None
    elif target == "unit":
        marker = "unit"
    elif target == "integration":
        marker = "integration"
    elif target == "e2e":
        marker = "e2e"
    elif target == "fast":
        marker = "not slow"
    elif target == "coverage":
        marker = None
    elif target == "auth":
        marker = "auth"
    elif target == "servers":
        marker = "servers"
    elif target == "search":
        marker = "search"
    elif target == "health":
        marker = "health"
    elif target == "core":
        marker = "core"
    else:
        raise ValueError(f"Unsupported test target: {target}")

    args = ["-m", marker] if marker else []
    return args


def _run_pytest(
    *,
    pytest_args: list[str],
) -> int:
    return subprocess.call(
        [sys.executable, "-m", "pytest", *pytest_args],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repo test suites via pytest markers.")
    parser.add_argument(
        "target",
        choices=[
            "check",
            "full",
            "unit",
            "integration",
            "e2e",
            "fast",
            "coverage",
            "auth",
            "servers",
            "search",
            "health",
            "core",
        ],
        help="Which test suite to run.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional args to pass through to pytest (prefix with `--`).",
    )
    args = parser.parse_args()

    if args.target == "check":
        return subprocess.call([sys.executable, "-m", "pytest", "--version"])

    try:
        selected_args = _build_pytest_args(args.target)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    passthrough = args.pytest_args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    return _run_pytest(
        pytest_args=[*selected_args, *passthrough],
    )


if __name__ == "__main__":
    raise SystemExit(main())

