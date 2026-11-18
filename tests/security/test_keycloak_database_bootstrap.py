"""Fail-closed restarts must preserve database volumes, not infer fresh installs."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "charts/keycloak/files/bootstrap.sh"


def run_bootstrap(data: Path, mode: str, **extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env={
            **os.environ,
            "PGDATA": str(data / "pgdata"),
            "BOOTSTRAP_MODE": mode,
            "KEYCLOAK_DB_USER": "keycloak",
            "KEYCLOAK_DB_NAME": "keycloak",
            **extra,
        },
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("mode", ["", "automatic", "false"])
def test_unknown_intent_never_initializes_database(tmp_path: Path, mode: str):
    result = run_bootstrap(tmp_path, mode)
    assert result.returncode != 0
    assert not (tmp_path / "pgdata").exists()


def test_existing_mode_refuses_empty_volume(tmp_path: Path):
    result = run_bootstrap(tmp_path, "existing", BOOTSTRAP_EXISTING_VERIFIED="true")
    assert result.returncode != 0
    assert "will not create one" in result.stderr
    assert not (tmp_path / "pgdata").exists()


@pytest.mark.parametrize("mode", ["fresh-install", "existing"])
def test_interrupted_restore_cannot_be_bypassed_by_mode_change(tmp_path: Path, mode: str):
    pending = tmp_path / ".keycloak-bootstrap-pending"
    pending.write_text("restore:" + "a" * 64 + "\n")
    cluster = tmp_path / "pgdata.bootstrap"
    cluster.mkdir()
    sentinel = cluster / "retained-data"
    sentinel.write_bytes(b"source data remains untouched")
    result = run_bootstrap(tmp_path, mode, BOOTSTRAP_EXISTING_VERIFIED="true")
    assert result.returncode != 0
    assert "different intent" in result.stderr
    assert pending.read_text() == "restore:" + "a" * 64 + "\n"
    assert sentinel.read_bytes() == b"source data remains untouched"


def test_interrupted_restore_cannot_switch_archives(tmp_path: Path):
    pending = tmp_path / ".keycloak-bootstrap-pending"
    pending.write_text("restore:" + "a" * 64 + "\n")
    result = run_bootstrap(
        tmp_path,
        "restore",
        BOOTSTRAP_SHA256="b" * 64,
        BOOTSTRAP_SOURCE_QUIESCED="true",
    )
    assert result.returncode != 0
    assert "different intent" in result.stderr
    assert pending.read_text() == "restore:" + "a" * 64 + "\n"


def test_pg_version_alone_never_proves_complete_initialization(tmp_path: Path):
    data = tmp_path / "pgdata"
    data.mkdir()
    version = data / "PG_VERSION"
    version.write_text("17\n")
    result = run_bootstrap(tmp_path, "fresh-install")
    assert result.returncode != 0
    assert "no completion receipt" in result.stderr
    assert version.read_text() == "17\n"


def test_partial_init_is_retained_not_reinitialized(tmp_path: Path):
    data = tmp_path / "pgdata"
    data.mkdir()
    sentinel = data / "partially-written-data"
    sentinel.write_text("keep")
    result = run_bootstrap(tmp_path, "fresh-install")
    assert result.returncode != 0
    assert "without PG_VERSION" in result.stderr
    assert sentinel.read_text() == "keep"
