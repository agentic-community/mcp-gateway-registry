"""
Unit tests for the EnforceAI dev bootstrap script.

These tests catch common footguns:
- token written in the wrong directory (compose vs local)
- token file not being a compact JWT (3 base64url parts)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_JWT_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")


def _run_bootstrap(
    *,
    cwd: Path,
    env: dict[str, str],
    args: list[str],
) -> None:
    subprocess.run(  # noqa: S603
        ["bash", "scripts/enforceai_dev_bootstrap.sh", *args],
        cwd=str(cwd),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _read_text(
    path: Path,
) -> str:
    return path.read_text().strip()

def _clean_enforceai_env(
    env: dict[str, str],
) -> dict[str, str]:
    cleaned = dict(env)

    # Developers commonly `source` the docker-compose env file, which sets
    # container paths like /app/enforceai_state/* on the host. The bootstrap script
    # intentionally honors env overrides, so tests must sanitize the environment.
    for key in list(cleaned.keys()):
        if key.startswith("ENFORCEAI_") or key == "OIDC_ISSUERS":
            cleaned.pop(key, None)

    return cleaned


@pytest.mark.unit
class TestEnforceAIDevBootstrapScript:
    def test_writes_valid_compact_jwt_to_enforceai_state_dir(
        self,
        tmp_path: Path,
    ) -> None:
        if shutil.which("openssl") is None:
            pytest.skip("bootstrap script requires openssl")

        repo_root = Path(__file__).resolve().parents[3]
        state_dir = tmp_path / "state"

        env = _clean_enforceai_env(os.environ.copy())
        env["ENFORCEAI_STATE_DIR"] = str(state_dir)
        env["ENFORCEAI_DB_PATH"] = str(state_dir / "enforceai.db")
        env["ENFORCEAI_SCOPES_CATALOG_PATH"] = str(repo_root / "auth_server" / "scopes.yml")
        env["ENFORCEAI_PYTHON"] = sys.executable

        _run_bootstrap(
            cwd=repo_root,
            env=env,
            args=["--force"],
        )

        token_path = state_dir / "bootstrap_gateway_token.txt"
        assert token_path.exists()

        token = _read_text(token_path)
        assert _JWT_RE.match(token)

        bearer_path = state_dir / "bootstrap_gateway_token.bearer.txt"
        assert bearer_path.exists()
        assert _read_text(bearer_path) == f"Bearer {token}"

    def test_compose_flag_writes_to_home_mcp_gateway_enforceai(
        self,
        tmp_path: Path,
    ) -> None:
        if shutil.which("openssl") is None:
            pytest.skip("bootstrap script requires openssl")

        repo_root = Path(__file__).resolve().parents[3]
        home_dir = tmp_path / "home"
        state_dir = home_dir / "mcp-gateway" / "enforceai"

        env = _clean_enforceai_env(os.environ.copy())
        env["HOME"] = str(home_dir)
        env["ENFORCEAI_PYTHON"] = sys.executable

        _run_bootstrap(
            cwd=repo_root,
            env=env,
            args=["--force", "--compose"],
        )

        token_path = state_dir / "bootstrap_gateway_token.txt"
        assert token_path.exists()

        token = _read_text(token_path)
        assert _JWT_RE.match(token)
