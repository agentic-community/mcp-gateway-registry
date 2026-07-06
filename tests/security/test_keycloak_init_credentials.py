"""
Regression tests for SA-9: Keycloak init scripts must not create users with weak
default passwords, must force a reset on first login (temporary: true), and must
not print credential values.

These are static-content assertions over the setup scripts and the realm import,
so they cannot regress even without a live Keycloak.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).parent.parent.parent


# Both copies of the Keycloak init script must satisfy the SA-9 invariants.
INIT_SCRIPTS = [
    "keycloak/setup/init-keycloak.sh",
    "terraform/aws-ecs/scripts/init-keycloak.sh",
]

# Weak default-password patterns that must never reappear in the init scripts.
FORBIDDEN_FALLBACKS = [
    ":-changeme",
    ":-testpass",
    ":-lob1pass",
    ":-lob2pass",
]


@pytest.mark.parametrize("script_path", INIT_SCRIPTS)
def test_init_script_has_no_weak_password_fallbacks(
    repo_root: Path,
    script_path: str,
):
    """No `${VAR:-weakdefault}` password fallbacks in the init scripts."""
    content = (repo_root / script_path).read_text()
    offenders = [pat for pat in FORBIDDEN_FALLBACKS if pat in content]
    assert not offenders, (
        f"{script_path}: weak default-password fallback(s) present: {offenders}. "
        "Require the password env var instead (fail closed when unset)."
    )


@pytest.mark.parametrize("script_path", INIT_SCRIPTS)
def test_init_script_creates_no_testuser(
    repo_root: Path,
    script_path: str,
):
    """The insecure `testuser` demo account must not be created."""
    content = (repo_root / script_path).read_text()
    assert "testuser" not in content, (
        f"{script_path}: still references 'testuser'; the weak demo account was removed."
    )


@pytest.mark.parametrize("script_path", INIT_SCRIPTS)
def test_init_script_users_are_temporary(
    repo_root: Path,
    script_path: str,
):
    """Every created credential forces a reset on first login (no temporary:false)."""
    content = (repo_root / script_path).read_text()
    assert '"temporary": false' not in content, (
        f"{script_path}: a user credential is created with 'temporary': false; "
        "all created users must use 'temporary': true to force a first-login reset."
    )


def test_primary_init_requires_admin_password(repo_root: Path):
    """The primary init script fails closed when INITIAL_ADMIN_PASSWORD is unset."""
    content = (repo_root / "keycloak/setup/init-keycloak.sh").read_text()
    assert 'if [ -z "$INITIAL_ADMIN_PASSWORD" ]' in content, (
        "keycloak/setup/init-keycloak.sh must guard on INITIAL_ADMIN_PASSWORD being set."
    )


def test_ecs_init_requires_lob_passwords(repo_root: Path):
    """The ECS init script fails closed when LOB user passwords are unset."""
    content = (repo_root / "terraform/aws-ecs/scripts/init-keycloak.sh").read_text()
    assert 'if [ -z "$LOB1_USER_PASSWORD" ] || [ -z "$LOB2_USER_PASSWORD" ]' in content, (
        "terraform/aws-ecs/scripts/init-keycloak.sh must guard on LOB user passwords."
    )


def test_realm_import_has_no_testuser(repo_root: Path):
    """The realm import must not ship a testuser with a weak default password."""
    data = json.loads((repo_root / "keycloak/import/realm-config.json").read_text())
    usernames = {u.get("username") for u in data.get("users", [])}
    assert "testuser" not in usernames, "realm-config.json still defines a 'testuser'."
    # Every remaining user credential must be temporary.
    for user in data.get("users", []):
        for cred in user.get("credentials", []):
            assert cred.get("temporary") is True, (
                f"realm-config.json user {user.get('username')!r} has a "
                "non-temporary credential; must be temporary: true."
            )
