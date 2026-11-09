"""
Tests for scripts/prepare-log-dirs.sh (Issue #1575).

On Linux the host log directory is owned by 1000:1000 to match the container's
appuser. On macOS, Docker Desktop accesses shared host directories as the macOS
user running Docker Desktop, so the directory must be owned by the invoking
user instead.

The script is run for real against a temp directory. ``uname``, ``sudo``,
``id``, ``chown`` and ``chmod`` are replaced by shims on PATH: ``uname``
reports the OS under test, ``sudo`` runs its arguments directly, ``id``
reports a typical macOS user (uid 501, gid 20 "staff") so it cannot be
mistaken for the Linux default of 1000, and ``chown``/``chmod`` record their
arguments instead of changing anything. This exercises the macOS branch on a
Linux CI runner.
"""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "prepare-log-dirs.sh"

# Identity reported by the ``id`` shim: a typical first macOS user.
MAC_USER_UID: str = "501"
MAC_USER_GID: str = "20"


def _write_shim(
    shim_dir: Path,
    name: str,
    body: str,
) -> None:
    shim = shim_dir / name
    shim.write_text(f"#!/usr/bin/env bash\n{body}\n")
    shim.chmod(0o755)


def _run_script(
    tmp_path: Path,
    os_name: str,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run the script with shims and return the result plus recorded calls."""
    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    calls_file = tmp_path / "calls.log"

    _write_shim(shim_dir, "uname", f'echo "{os_name}"')
    _write_shim(shim_dir, "sudo", 'exec "$@"')
    _write_shim(
        shim_dir,
        "id",
        f'if [ "$1" = "-g" ]; then echo {MAC_USER_GID}; else echo {MAC_USER_UID}; fi',
    )
    _write_shim(shim_dir, "chown", f'echo "chown $*" >> "{calls_file}"')
    _write_shim(shim_dir, "chmod", f'echo "chmod $*" >> "{calls_file}"')

    env = {
        "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        "APP_LOG_DIR": str(tmp_path / "logs"),
    }
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],  # nosec B603 B607 - repo script under test
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    calls = calls_file.read_text().splitlines() if calls_file.exists() else []
    return result, calls


class TestPrepareLogDirs:
    """Ownership and mode chosen by prepare-log-dirs.sh per host OS."""

    def test_linux_owner_is_container_uid(
        self,
        tmp_path: Path,
    ) -> None:
        result, calls = _run_script(tmp_path, "Linux")

        assert result.returncode == 0, result.stderr
        assert f"chown -R 1000:1000 {tmp_path / 'logs'}" in calls

    def test_macos_owner_is_invoking_user(
        self,
        tmp_path: Path,
    ) -> None:
        result, calls = _run_script(tmp_path, "Darwin")

        assert result.returncode == 0, result.stderr
        expected = f"chown -R {MAC_USER_UID}:{MAC_USER_GID} {tmp_path / 'logs'}"
        assert expected in calls

    def test_macos_under_sudo_uses_sudo_caller(
        self,
        tmp_path: Path,
    ) -> None:
        result, calls = _run_script(
            tmp_path,
            "Darwin",
            extra_env={"SUDO_UID": "502", "SUDO_GID": "21"},
        )

        assert result.returncode == 0, result.stderr
        assert f"chown -R 502:21 {tmp_path / 'logs'}" in calls

    @pytest.mark.parametrize("os_name", ["Linux", "Darwin"])
    def test_explicit_owner_override_wins(
        self,
        tmp_path: Path,
        os_name: str,
    ) -> None:
        result, calls = _run_script(
            tmp_path,
            os_name,
            extra_env={"LOG_DIR_OWNER_UID": "2000", "LOG_DIR_OWNER_GID": "3000"},
        )

        assert result.returncode == 0, result.stderr
        assert f"chown -R 2000:3000 {tmp_path / 'logs'}" in calls

    @pytest.mark.parametrize("os_name", ["Linux", "Darwin"])
    def test_mode_stays_0750_by_default(
        self,
        tmp_path: Path,
        os_name: str,
    ) -> None:
        result, calls = _run_script(tmp_path, os_name)

        assert result.returncode == 0, result.stderr
        assert f"chmod 0750 {tmp_path / 'logs'}" in calls
