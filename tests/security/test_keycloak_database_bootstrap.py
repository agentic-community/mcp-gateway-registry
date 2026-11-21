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


@pytest.mark.parametrize("parameter", ["KEYCLOAK_DB_USER", "KEYCLOAK_DB_NAME"])
@pytest.mark.parametrize(
    "value",
    [
        "",
        "a" * 64,
        "7identity",
        "-identity",
        " identity",
        "identity ",
        "identity.name",
        "identity/name",
        "identity:name",
        "identity\n",
        "identity\t",
        "identit\u00e9",
        "a" * 62 + "\u00e9",
        "identity' OR true; --",
        'identity"',
        "$(touch injected)",
    ],
)
def test_invalid_names_refuse_before_creating_or_changing_data(tmp_path, parameter, value):
    unused = tmp_path / "unused"
    result = run_bootstrap(unused, "fresh-install", **{parameter: value})
    assert result.returncode != 0
    assert f"{parameter} must be 1-63 ASCII characters" in result.stderr
    assert not unused.exists()

    data = tmp_path / "pgdata"
    data.mkdir()
    (data / "PG_VERSION").write_text("17\n")
    (data / ".keycloak-bootstrap-complete").write_text("fresh-install:\n")
    (data / "retained-data").write_bytes(b"original database bytes")
    pending = tmp_path / ".keycloak-bootstrap-pending"
    pending.write_text("fresh-install:\n")
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    result = run_bootstrap(tmp_path, "fresh-install", **{parameter: value})
    assert result.returncode != 0
    assert parameter in result.stderr
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("KEYCLOAK_DB_USER", "postgres"),
        ("KEYCLOAK_DB_USER", "PoStGrEs"),
        ("KEYCLOAK_DB_USER", "public"),
        ("KEYCLOAK_DB_USER", "PuBlIc"),
        ("KEYCLOAK_DB_USER", "pg_"),
        ("KEYCLOAK_DB_USER", "PG_identity"),
        ("KEYCLOAK_DB_NAME", "postgres"),
        ("KEYCLOAK_DB_NAME", "POSTGRES"),
        ("KEYCLOAK_DB_NAME", "template0"),
        ("KEYCLOAK_DB_NAME", "Template0"),
        ("KEYCLOAK_DB_NAME", "template1"),
        ("KEYCLOAK_DB_NAME", "TEMPLATE1"),
    ],
)
def test_reserved_names_fail_before_volume_mutation(tmp_path, parameter, value):
    root = tmp_path / "unused"
    result = run_bootstrap(root, "fresh-install", **{parameter: value})
    assert result.returncode != 0
    assert f"{parameter} must not" in result.stderr
    assert not root.exists()
    root.mkdir()
    data = root / "pgdata"
    data.mkdir()
    (data / "PG_VERSION").write_text("17\n")
    (data / ".keycloak-bootstrap-complete").write_text("fresh-install:\n")
    sentinel = data / "retained-data"
    sentinel.write_bytes(b"original database bytes")
    result = run_bootstrap(root, "fresh-install", **{parameter: value})
    assert result.returncode != 0
    assert f"{parameter} must not" in result.stderr
    assert sentinel.read_bytes() == b"original database bytes"
    assert (data / "PG_VERSION").read_text() == "17\n"
    assert (data / ".keycloak-bootstrap-complete").read_text() == "fresh-install:\n"


@pytest.mark.parametrize("parameter", ["KEYCLOAK_DB_USER", "KEYCLOAK_DB_NAME"])
@pytest.mark.parametrize(
    "value", ["a", "_", "a" * 63, "Identity_Store-7", "pg-identity", "postgres2", "template2"]
)
def test_valid_name_boundaries_reach_existing_volume_safety_check(tmp_path, parameter, value):
    result = run_bootstrap(
        tmp_path, "existing", BOOTSTRAP_EXISTING_VERIFIED="true", **{parameter: value}
    )
    assert result.returncode != 0
    assert "will not create one" in result.stderr
    assert not (tmp_path / "pgdata").exists()


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("KEYCLOAK_DB_USER", "template0"), ("KEYCLOAK_DB_NAME", "pg_identity")],
)
def test_reserved_names_apply_only_to_their_parameter(tmp_path, parameter, value):
    result = run_bootstrap(
        tmp_path, "existing", BOOTSTRAP_EXISTING_VERIFIED="true", **{parameter: value}
    )
    assert result.returncode != 0
    assert "will not create one" in result.stderr
    assert not (tmp_path / "pgdata").exists()


@pytest.mark.parametrize(
    ("marker", "error"),
    [
        (None, "staging database has no identity marker"),
        ("Other_Owner-8\nkeycloak\n", "staging database has a different identity"),
        ("keycloak\nOther_Store-8\n", "staging database has a different identity"),
        ("keycloak\nkeycloak\n\n", "staging database has a different identity"),
    ],
)
def test_staged_identity_must_match_before_any_database_access(tmp_path, marker, error):
    staging = tmp_path / "pgdata.bootstrap"
    staging.mkdir()
    (staging / "PG_VERSION").write_text("17\n")
    (staging / "retained-data").write_bytes(b"retained incomplete restore")
    pending = tmp_path / ".keycloak-bootstrap-pending"
    pending.write_text("restore:" + "a" * 64 + "\n")
    if marker is not None:
        (staging / ".keycloak-bootstrap-identity").write_text(marker)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    result = run_bootstrap(
        tmp_path, "restore", BOOTSTRAP_SHA256="a" * 64, BOOTSTRAP_SOURCE_QUIESCED="true"
    )
    assert result.returncode != 0
    assert error in result.stderr
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


def test_initializing_identity_cannot_change_before_staging_promotion(tmp_path):
    initializing = tmp_path / "pgdata.initializing"
    initializing.mkdir()
    marker = initializing / ".keycloak-bootstrap-identity"
    marker.write_text("Realm_Owner-7\nIdentity_Store-7\n")
    result = run_bootstrap(tmp_path, "fresh-install")
    assert result.returncode != 0
    assert "initializing database has a different identity" in result.stderr
    assert marker.read_text() == "Realm_Owner-7\nIdentity_Store-7\n"
    assert not (tmp_path / "pgdata").exists()
    assert not (tmp_path / "pgdata.bootstrap").exists()
