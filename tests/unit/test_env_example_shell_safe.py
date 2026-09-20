"""`.env.example` must be safe for `source` to read.

The documented first step of a local deployment is `cp .env.example .env`, and
`build_and_run.sh` then does `source .env` (four call sites). So the shell, not a
dotenv parser, is what reads this file first. A value the shell mis-parses breaks
the very first command a new user runs.

This caught a real one: `CIMD_CLIENT_NAME=AI Registry Tools` (PR #1711,
unquoted). bash read it as "assign CIMD_CLIENT_NAME=AI, then run the command
`Registry` with argument `Tools`", so `./build_and_run.sh` died with
``.env: line 1408: Registry: command not found``. Had it survived, the variable
would have held ``AI`` rather than the full name, which is the quieter half of
the bug: a truncated value that looks plausible.

Every other space-containing value in the file was already quoted
(``REGISTRY_NAME="AI Gateway Registry"``, ``REGISTRY_ORGANIZATION_NAME="ACME
Inc."``), so the convention existed and one line missed it. These tests make the
convention enforceable instead of remembered.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"

# KEY=VALUE, tolerating a leading `export`. Captures the raw right-hand side
# exactly as the shell would see it, before any quote handling.
ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _assignments() -> list[tuple[int, str, str]]:
    """Every (line number, key, raw value) assignment in .env.example."""
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(ENV_EXAMPLE.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ASSIGNMENT.match(line)
        if match:
            out.append((lineno, match.group(1), match.group(2)))
    return out


def _strip_trailing_comment(raw: str) -> str:
    """Drop a ` # ...` trailing comment the shell would also drop.

    Only applies outside quotes: bash treats ``#`` after whitespace as starting a
    comment, so ``KEY=value # note`` assigns ``value``. A ``#`` inside quotes or
    with no preceding space is part of the value.
    """
    if raw[:1] in {'"', "'"}:
        return raw
    return re.split(r"\s+#", raw, maxsplit=1)[0]


def test_env_example_exists() -> None:
    assert ENV_EXAMPLE.is_file(), f"{ENV_EXAMPLE} is missing"


def test_every_value_with_whitespace_is_quoted() -> None:
    """An unquoted value containing whitespace is a shell command, not a value.

    This is the precise diagnostic: it names the line and key so the fix is
    obvious, where the source test below only reports that something broke.
    """
    offenders = []
    for lineno, key, raw in _assignments():
        value = _strip_trailing_comment(raw).strip()
        if not value or value[0] in {'"', "'"}:
            continue
        if re.search(r"\s", value):
            offenders.append(f"  .env.example:{lineno}  {key}={value}")

    assert not offenders, (
        "Unquoted value(s) containing whitespace. `source .env` would split "
        "these into an assignment plus a command, truncating the value and "
        "failing build_and_run.sh. Wrap the value in double quotes:\n" + "\n".join(offenders)
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_env_example_can_be_sourced_by_bash() -> None:
    """Catch the whole class, not just the whitespace case.

    Backticks, ``$(...)``, unbalanced quotes and stray metacharacters all break
    `source` in ways the targeted check above would miss.
    """
    result = subprocess.run(  # nosec B603 B607 - fixed argv, repo-local file
        ["bash", "-c", f'set -a; source "{ENV_EXAMPLE}"; set +a'],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"`source .env.example` failed (exit {result.returncode}). A fresh "
        f"`cp .env.example .env` would break build_and_run.sh:\n{result.stderr.strip()}"
    )
    assert not result.stderr.strip(), (
        f"`source .env.example` wrote to stderr:\n{result.stderr.strip()}"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_sourced_values_are_not_truncated() -> None:
    """The quiet half of the bug: a value that parses but loses its tail.

    `CIMD_CLIENT_NAME=AI Registry Tools` sources "successfully" in some shells
    while assigning only ``AI``. Comparing what bash produced against what the
    file declares catches that, where an exit-code check alone does not.
    """
    declared = {
        key: _strip_trailing_comment(raw).strip().strip("\"'") for _, key, raw in _assignments()
    }
    # Only values with whitespace can be truncated this way, and only non-empty
    # ones are worth comparing.
    interesting = {k: v for k, v in declared.items() if v and re.search(r"\s", v)}
    if not interesting:
        pytest.skip("no multi-word values in .env.example")

    probe = "; ".join(f'printf "%s=%s\\n" {k} "${k}"' for k in interesting)
    result = subprocess.run(  # nosec B603 B607 - fixed argv, keys are [A-Za-z0-9_]
        ["bash", "-c", f'set -a; source "{ENV_EXAMPLE}"; set +a; {probe}'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.strip()

    actual = dict(line.split("=", 1) for line in result.stdout.strip().splitlines() if "=" in line)
    mismatched = [
        f"  {key}: file says {want!r}, shell got {actual.get(key)!r}"
        for key, want in interesting.items()
        if actual.get(key) != want
    ]
    assert not mismatched, (
        "Value(s) changed when bash sourced the file, which means the shell "
        "re-interpreted them:\n" + "\n".join(mismatched)
    )
