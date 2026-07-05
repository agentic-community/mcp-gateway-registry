"""
Regression tests for SA-11: the default MongoDB in docker-compose must run with
authentication enabled and must not ship a weak default password.

Static-content assertions over docker-compose.yml and .env.example, so they hold
without a live MongoDB.
"""

import re
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).parent.parent.parent


def test_default_mongodb_runs_with_auth(repo_root: Path):
    """The default docker-compose.yml mongodb must enable --auth and a keyFile."""
    content = (repo_root / "docker-compose.yml").read_text()
    # Locate the mongodb service command line.
    cmd_match = re.search(r"^\s*command:\s*mongod .*$", content, re.MULTILINE)
    assert cmd_match, "Could not find the mongod command in docker-compose.yml"
    joined = "\n".join(
        line for line in content.splitlines() if "mongod" in line and "--replSet" in line
    )
    assert "--auth" in joined, "docker-compose.yml mongodb must run with --auth"
    assert "--keyFile" in joined, "docker-compose.yml mongodb must run with --keyFile (replica-set auth)"


def test_default_mongodb_has_keyfile_init(repo_root: Path):
    """A keyfile-init helper must exist to generate the replica-set keyfile."""
    content = (repo_root / "docker-compose.yml").read_text()
    assert "mongodb-keyfile-init:" in content, (
        "docker-compose.yml must define mongodb-keyfile-init to generate the keyfile"
    )
    assert "mongodb-keyfile:" in content, "docker-compose.yml must define the mongodb-keyfile volume"


def test_mongodb_credentials_fail_closed(repo_root: Path):
    """Root creds must use the required-form ${VAR:?...}, never a weak default."""
    content = (repo_root / "docker-compose.yml").read_text()
    assert "MONGO_INITDB_ROOT_USERNAME=${DOCUMENTDB_USERNAME:?" in content, (
        "MONGO_INITDB_ROOT_USERNAME must fail closed on unset DOCUMENTDB_USERNAME"
    )
    assert "MONGO_INITDB_ROOT_PASSWORD=${DOCUMENTDB_PASSWORD:?" in content, (
        "MONGO_INITDB_ROOT_PASSWORD must fail closed on unset DOCUMENTDB_PASSWORD"
    )
    # No weak admin/admin fallback anywhere for the mongo root password.
    assert "DOCUMENTDB_PASSWORD:-admin" not in content, (
        "docker-compose.yml must not fall back to the weak 'admin' password"
    )


def test_env_example_has_no_admin_password_default(repo_root: Path):
    """.env.example must not ship an active DOCUMENTDB_PASSWORD=admin assignment."""
    content = (repo_root / ".env.example").read_text()
    # Any active (uncommented) DOCUMENTDB_PASSWORD line must not be 'admin'.
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("DOCUMENTDB_PASSWORD="):
            value = stripped.split("=", 1)[1].strip()
            assert value != "admin", (
                ".env.example ships DOCUMENTDB_PASSWORD=admin; must be blank or a placeholder"
            )
