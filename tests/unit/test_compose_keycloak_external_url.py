"""Compose parity check for KEYCLOAK_EXTERNAL_URL.

Both the registry and the auth server build OAuth discovery documents
(`.well-known/oauth-authorization-server`) from KEYCLOAK_EXTERNAL_URL. When the
variable is absent from a service's environment, `_create_keycloak_provider`
falls back to KEYCLOAK_URL, so that service advertises the internal Docker DNS
name. The response is a valid metadata document, nothing errors server-side, and
external MCP clients fail OAuth discovery with a generic error.

The variable has gone missing from a service before (issue #1649, and #1378 for
the Podman stack), so this asserts the invariant across every compose file and
every service that serves discovery metadata.
"""

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

COMPOSE_FILES = [
    "docker-compose.yml",
    "docker-compose.prebuilt.yml",
    "docker-compose.podman.yml",
]

# Services running an application that serves OAuth discovery metadata.
DISCOVERY_SERVICES = ["registry", "auth-server"]


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Repository root directory."""
    return Path(__file__).parent.parent.parent


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
@pytest.mark.parametrize("service", DISCOVERY_SERVICES)
def test_discovery_service_receives_keycloak_external_url(
    repo_root: Path,
    compose_filename: str,
    service: str,
):
    """Every service serving OAuth metadata must receive KEYCLOAK_EXTERNAL_URL.

    Without it the service advertises the internal KEYCLOAK_URL, which no
    external client can resolve, and the failure is silent server-side.
    """
    env = _service_environment(repo_root / compose_filename, service)

    assert "KEYCLOAK_EXTERNAL_URL" in env, (
        f"{compose_filename}: service '{service}' serves OAuth discovery metadata but does "
        f"not receive KEYCLOAK_EXTERNAL_URL, so it will advertise the internal "
        f"KEYCLOAK_URL that external MCP clients cannot reach."
    )


@pytest.mark.parametrize("compose_filename", COMPOSE_FILES)
@pytest.mark.parametrize("service", DISCOVERY_SERVICES)
def test_keycloak_external_url_is_operator_overridable(
    repo_root: Path,
    compose_filename: str,
    service: str,
):
    """The value must interpolate ``${KEYCLOAK_EXTERNAL_URL}`` from the operator's .env.

    Hard-coding it would silently ignore the setting an operator puts in .env,
    which is the same outage with a harder-to-find cause.
    """
    env = _service_environment(repo_root / compose_filename, service)
    value = env["KEYCLOAK_EXTERNAL_URL"]

    assert value.startswith("${KEYCLOAK_EXTERNAL_URL"), (
        f"{compose_filename}: service '{service}' sets KEYCLOAK_EXTERNAL_URL to "
        f"'{value}' instead of interpolating the operator's value from .env."
    )


@pytest.mark.parametrize("compose_filename", COMPOSE_FILES)
@pytest.mark.parametrize("service", DISCOVERY_SERVICES)
def test_internal_and_external_urls_are_both_present(
    repo_root: Path,
    compose_filename: str,
    service: str,
):
    """KEYCLOAK_URL must remain alongside it: the two serve different traffic.

    KEYCLOAK_URL is the container-to-container hop; KEYCLOAK_EXTERNAL_URL is what
    leaves the deployment. Replacing one with the other breaks the opposite half.
    """
    env = _service_environment(repo_root / compose_filename, service)

    assert "KEYCLOAK_URL" in env, (
        f"{compose_filename}: service '{service}' is missing KEYCLOAK_URL, which is the "
        f"URL used for container-to-container traffic."
    )
