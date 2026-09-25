"""The realm registry authenticates its admin user against must be configurable.

`registry/utils/keycloak_manager.py` calls the Keycloak Admin REST API with the
password grant, against one specific realm's token endpoint. The realm in that
URL is where the *admin user* lives, which is not the realm being administered
(`KEYCLOAK_REALM`). The two coincide only in the default deployment, where the
admin is a `master` user holding roles on the `mcp-gateway-realm` client.

An operator who would rather not distribute a `master` credential grants the
same `realm-management` roles to a user inside the managed realm. That user
cannot authenticate while the token endpoint is pinned to `master`: Keycloak
isolates users per realm, so the master token endpoint answers `invalid_grant`
for a user that lives elsewhere. The failure is a generic admin-API error with
nothing in it to suggest the realm is the cause.

The compose assertions below cover the other half of the same path — the value
has to reach the registry container before the setting can take effect.
"""

import importlib
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

COMPOSE_FILES = [
    "docker-compose.yml",
    "docker-compose.prebuilt.yml",
    "docker-compose.podman.yml",
]

# Only the registry calls the admin token endpoint; the auth server does not use
# keycloak_manager.
ADMIN_SERVICE = "registry"

MODULE_NAME = "registry.utils.keycloak_manager"

DEFAULT_ADMIN_REALM = "master"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def repo_root() -> Path:
    """Repository root directory."""
    return Path(__file__).parent.parent.parent


class _RecordingResponse:
    """Minimal httpx.Response stand-in that hands back a token."""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"access_token": "test-admin-token"}


class _RecordingClient:
    """Minimal httpx.AsyncClient stand-in that records requested URLs."""

    def __init__(self, recorded: list[str], **_kwargs: Any) -> None:
        self._recorded = recorded

    async def __aenter__(self) -> "_RecordingClient":
        return self

    async def __aexit__(self, *_exc_info: Any) -> bool:
        return False

    async def post(self, url: str, **_kwargs: Any) -> _RecordingResponse:
        self._recorded.append(url)
        return _RecordingResponse()


@pytest.fixture
def admin_token_url(monkeypatch):
    """Reload keycloak_manager under the caller's env and capture its token URL.

    The module reads its configuration into module-level constants at import
    time, so a reload is what makes an env change visible.
    """

    async def _capture(**env: str) -> str:
        monkeypatch.setenv("KEYCLOAK_ADMIN_PASSWORD", "test-admin-password")
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        module = importlib.reload(importlib.import_module(MODULE_NAME))

        recorded: list[str] = []
        monkeypatch.setattr(
            module.httpx, "AsyncClient", lambda **kw: _RecordingClient(recorded, **kw)
        )

        token = await module._get_keycloak_admin_token()

        assert token == "test-admin-token"
        assert len(recorded) == 1
        return recorded[0]

    yield _capture

    # The reload above bakes this test's env into the module. Undo the env and
    # reload once more so the next test starts from its own environment.
    monkeypatch.undo()
    importlib.reload(importlib.import_module(MODULE_NAME))


# =============================================================================
# UNITS: REALM RESOLUTION
# =============================================================================


async def test_defaults_to_master_when_unset(
    monkeypatch,
    admin_token_url,
) -> None:
    """An unconfigured deployment must keep authenticating exactly as before."""
    monkeypatch.delenv("KEYCLOAK_ADMIN_REALM", raising=False)

    url = await admin_token_url()

    assert url == "http://keycloak:8080/realms/master/protocol/openid-connect/token"


async def test_empty_value_falls_back_to_master(
    admin_token_url,
) -> None:
    """An empty string in .env must not produce a ``//realms//`` token URL."""
    url = await admin_token_url(KEYCLOAK_ADMIN_REALM="")

    assert "/realms/master/" in url


async def test_honours_configured_realm(
    admin_token_url,
) -> None:
    """A realm-internal admin user requires the token endpoint to follow it."""
    url = await admin_token_url(KEYCLOAK_ADMIN_REALM="mcp-gateway")

    assert url == "http://keycloak:8080/realms/mcp-gateway/protocol/openid-connect/token"


async def test_ignores_keycloak_realm(
    admin_token_url,
) -> None:
    """KEYCLOAK_REALM names the administered realm, not the admin user's realm.

    Deriving the token realm from it would break every deployment whose admin
    lives in `master` while the managed realm is something else.
    """
    url = await admin_token_url(KEYCLOAK_REALM="managed-realm")

    assert "/realms/master/" in url
    assert "/realms/managed-realm/" not in url


# =============================================================================
# COMPOSE: THE VALUE MUST REACH THE CONTAINER
# =============================================================================


def _service_environment(compose_path: Path, service: str) -> dict[str, str]:
    """Return a service's environment as a mapping.

    Compose accepts both the ``KEY: value`` mapping form and the ``- KEY=value``
    list form; normalise to a dict so assertions work against either.
    """
    data = yaml.safe_load(compose_path.read_text())
    env = data["services"][service].get("environment", {}) or {}
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    normalised: dict[str, str] = {}
    for entry in env:
        key, _, value = str(entry).partition("=")
        normalised[key] = value
    return normalised


@pytest.mark.parametrize("compose_filename", COMPOSE_FILES)
def test_registry_receives_admin_realm(
    repo_root: Path,
    compose_filename: str,
) -> None:
    """Setting KEYCLOAK_ADMIN_REALM in .env must be able to reach the registry.

    Compose passes an allow-list of variables, so a variable the registry reads
    but compose omits is unset inside the container and the default silently
    wins — the operator sees `invalid_grant` despite having set it correctly.
    """
    env = _service_environment(repo_root / compose_filename, ADMIN_SERVICE)

    assert "KEYCLOAK_ADMIN_REALM" in env, (
        f"{compose_filename}: service '{ADMIN_SERVICE}' does not receive "
        f"KEYCLOAK_ADMIN_REALM, so the setting in .env never reaches the container "
        f"and the admin token endpoint stays pinned to '{DEFAULT_ADMIN_REALM}'."
    )


@pytest.mark.parametrize("compose_filename", COMPOSE_FILES)
def test_admin_realm_is_operator_overridable(
    repo_root: Path,
    compose_filename: str,
) -> None:
    """The value must interpolate from .env and default to master."""
    env = _service_environment(repo_root / compose_filename, ADMIN_SERVICE)
    value = env["KEYCLOAK_ADMIN_REALM"]

    assert value.startswith("${KEYCLOAK_ADMIN_REALM"), (
        f"{compose_filename}: service '{ADMIN_SERVICE}' sets KEYCLOAK_ADMIN_REALM to "
        f"'{value}' instead of interpolating the operator's value from .env."
    )
    assert value.endswith(f":-{DEFAULT_ADMIN_REALM}}}"), (
        f"{compose_filename}: KEYCLOAK_ADMIN_REALM must default to "
        f"'{DEFAULT_ADMIN_REALM}' so existing deployments are unaffected."
    )


@pytest.mark.parametrize("compose_filename", COMPOSE_FILES)
def test_keycloak_realm_stays_separate(
    repo_root: Path,
    compose_filename: str,
) -> None:
    """KEYCLOAK_REALM must remain independently present.

    Collapsing the two into one variable would make the admin realm
    unconfigurable for any deployment where they differ.
    """
    env = _service_environment(repo_root / compose_filename, ADMIN_SERVICE)

    assert "KEYCLOAK_REALM" in env
    assert env["KEYCLOAK_REALM"] != env["KEYCLOAK_ADMIN_REALM"]


# =============================================================================
# REPO-WIDE GUARD: NO NEW master-REALM HARDCODES
# =============================================================================

# Provisioning scripts, charts and terraform used to curl
# `/realms/master/protocol/openid-connect/token` directly, so setting
# KEYCLOAK_ADMIN_REALM fixed the registry while every script kept hitting
# master. Each of them now resolves `${KEYCLOAK_ADMIN_REALM:-master}` at call
# time instead; this scan keeps the hardcode from coming back.

_SKIP_DIRS = {".claude", "docs"}

_HARDCODE_PATTERNS = ("realms/master", "--realm master")


def test_no_master_realm_hardcodes_outside_docs(repo_root: Path) -> None:
    """No executable surface may pin the admin realm to master.

    Docs are exempt: their snippets illustrate the default deployment, where
    the admin realm is master. Everything else — scripts, charts, terraform —
    must resolve KEYCLOAK_ADMIN_REALM so code and deployment cannot drift.

    Only git-tracked files are scanned, so test-run artifacts (coverage
    reports, __pycache__) cannot trip the guard.
    """
    tracked = subprocess.run(
        ["git", "ls-files"],  # nosec B603 B607 - hardcoded command
        capture_output=True,
        text=True,
        timeout=30,
        cwd=repo_root,
        check=True,
    ).stdout.splitlines()

    offenders: list[str] = []
    for rel in tracked:
        path = repo_root / rel
        if _SKIP_DIRS & set(Path(rel).parts) or rel == str(Path(__file__).relative_to(repo_root)):
            continue
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(pattern in line for pattern in _HARDCODE_PATTERNS):
                offenders.append(f"{rel}:{line_no}")

    assert not offenders, (
        "master-realm hardcodes found outside docs/; resolve "
        "KEYCLOAK_ADMIN_REALM instead:\n" + "\n".join(offenders)
    )
