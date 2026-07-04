"""
Unit tests for Docker container security configuration.

Tests verify that Dockerfiles follow CIS Docker Benchmark 4.1 requirements:
- Non-root USER directive
- No sudo package
- HEALTHCHECK directives
- Proper environment variables (PIP_NO_CACHE_DIR)

Also verifies the compose port-exposure hardening for SA-5: every published
port except the nginx front door (80/443) must be bound to a loopback-by-default
host interface (``${HOST_BIND_IP:-127.0.0.1}``) so datastores, the vault, admin
consoles and backend MCP servers are never exposed on all interfaces out of the
box.
"""

import re
from pathlib import Path

import pytest
import yaml

# Compose files that must follow the loopback-bind invariant.
COMPOSE_FILES = [
    "docker-compose.yml",
    "docker-compose.prebuilt.yml",
    "docker-compose.podman.yml",
]

# The only host-published container ports allowed to bind all interfaces: the
# nginx front door. Keyed by container-side target port.
FRONT_DOOR_TARGET_PORTS = {8080, 8443}

# Expected loopback-default host-bind expression for every other published port.
LOOPBACK_BIND_PREFIX = "${HOST_BIND_IP:-127.0.0.1}:"

# List of Dockerfiles to test
DOCKERFILES = [
    "Dockerfile",
    "docker/Dockerfile.auth",
    "docker/Dockerfile.registry",
    "docker/Dockerfile.mcp-server",
    "docker/Dockerfile.mcp-server-light",
    "docker/Dockerfile.metrics-db",
    "docker/keycloak/Dockerfile",
    "metrics-service/Dockerfile",
    "terraform/aws-ecs/grafana/Dockerfile",
]


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).parent.parent.parent


@pytest.mark.parametrize("dockerfile_path", DOCKERFILES)
def test_dockerfile_has_user_directive(repo_root: Path, dockerfile_path: str):
    """Test that Dockerfile has USER directive (CIS Docker Benchmark 4.1)."""
    dockerfile = repo_root / dockerfile_path
    assert dockerfile.exists(), f"Dockerfile not found: {dockerfile}"

    content = dockerfile.read_text()

    # Check for USER directive
    user_pattern = re.compile(r"^USER\s+\w+", re.MULTILINE)
    assert user_pattern.search(content), f"{dockerfile_path}: Missing USER directive (CIS 4.1)"


@pytest.mark.parametrize("dockerfile_path", DOCKERFILES)
def test_dockerfile_user_not_root(repo_root: Path, dockerfile_path: str):
    """Test that Dockerfile does not run as root user."""
    dockerfile = repo_root / dockerfile_path
    assert dockerfile.exists(), f"Dockerfile not found: {dockerfile}"

    content = dockerfile.read_text()

    # Find all USER directives
    user_lines = re.findall(r"^USER\s+(\w+)", content, re.MULTILINE)
    assert user_lines, f"{dockerfile_path}: No USER directive found"

    # Last USER directive should not be root
    last_user = user_lines[-1]
    assert last_user.lower() != "root", f"{dockerfile_path}: Last USER directive is 'root'"


@pytest.mark.parametrize("dockerfile_path", DOCKERFILES)
def test_dockerfile_no_sudo(repo_root: Path, dockerfile_path: str):
    """Test that Dockerfile does not install sudo package."""
    dockerfile = repo_root / dockerfile_path
    assert dockerfile.exists(), f"Dockerfile not found: {dockerfile}"

    content = dockerfile.read_text()

    # Check that sudo is not being installed
    assert "sudo" not in content, f"{dockerfile_path}: Contains 'sudo' package (security risk)"


@pytest.mark.parametrize("dockerfile_path", DOCKERFILES)
def test_dockerfile_has_healthcheck(repo_root: Path, dockerfile_path: str):
    """Test that Dockerfile has HEALTHCHECK directive."""
    dockerfile = repo_root / dockerfile_path
    assert dockerfile.exists(), f"Dockerfile not found: {dockerfile}"

    content = dockerfile.read_text()

    # Check for HEALTHCHECK directive
    healthcheck_pattern = re.compile(r"^HEALTHCHECK\s+", re.MULTILINE)
    assert healthcheck_pattern.search(content), f"{dockerfile_path}: Missing HEALTHCHECK directive"


@pytest.mark.parametrize(
    "dockerfile_path",
    [
        f
        for f in DOCKERFILES
        if not f.startswith("terraform/")  # Exclude Grafana (Node.js)
        and not f.endswith("metrics-db")  # Exclude alpine-based
    ],
)
def test_python_dockerfile_has_pip_no_cache(repo_root: Path, dockerfile_path: str):
    """Test that Python Dockerfiles set PIP_NO_CACHE_DIR=1."""
    dockerfile = repo_root / dockerfile_path
    assert dockerfile.exists(), f"Dockerfile not found: {dockerfile}"

    content = dockerfile.read_text()

    # Check if it's a Python-based image
    if re.search(r"FROM.*python", content, re.IGNORECASE):
        # Check for PIP_NO_CACHE_DIR
        assert "PIP_NO_CACHE_DIR" in content, (
            f"{dockerfile_path}: Python image missing PIP_NO_CACHE_DIR"
        )


def test_docker_compose_has_security_options(repo_root: Path):
    """Test that docker-compose.yml has security hardening options."""
    compose_file = repo_root / "docker-compose.yml"
    assert compose_file.exists(), "docker-compose.yml not found"

    content = compose_file.read_text()

    # Check for security_opt
    assert "security_opt:" in content, "docker-compose.yml missing security_opt"
    assert "no-new-privileges:true" in content, "docker-compose.yml missing no-new-privileges"

    # Check for cap_drop
    assert "cap_drop:" in content, "docker-compose.yml missing cap_drop"
    assert "- ALL" in content, "docker-compose.yml missing cap_drop: ALL"


def test_docker_compose_mongodb_cap_add(repo_root: Path):
    """Test that all docker-compose files restore SETUID/SETGID for MongoDB after cap_drop ALL.

    MongoDB uses gosu to switch from root to the mongodb user at startup.
    gosu requires SETUID and SETGID capabilities. Without them, MongoDB
    fails with: 'error: failed switching to mongodb: operation not permitted'.

    Regression introduced in PR #624 and PR #651 where cap_drop: ALL was applied
    to all services without adding back the minimum capabilities required by MongoDB.
    Fixed in PR #688.
    """
    compose_files = [
        "docker-compose.yml",
        "docker-compose.prebuilt.yml",
        "docker-compose.podman.yml",
    ]
    for compose_filename in compose_files:
        compose_file = repo_root / compose_filename
        assert compose_file.exists(), f"{compose_filename} not found"

        content = compose_file.read_text()

        assert "cap_add:" in content, f"{compose_filename}: missing cap_add for MongoDB"
        assert "- SETUID" in content, (
            f"{compose_filename}: missing SETUID in cap_add (required by MongoDB gosu)"
        )
        assert "- SETGID" in content, (
            f"{compose_filename}: missing SETGID in cap_add (required by MongoDB gosu)"
        )


def test_docker_compose_registry_port_mapping(repo_root: Path):
    """Test that docker-compose.yml maps nginx to high ports."""
    compose_file = repo_root / "docker-compose.yml"
    assert compose_file.exists(), "docker-compose.yml not found"

    content = compose_file.read_text()

    # Check for port mapping 80:8080 and 443:8443 (front door, bound on all
    # interfaces; the loopback-bind invariant tests below exempt these).
    assert "80:8080" in content, "Missing port mapping 80:8080"
    assert "443:8443" in content, "Missing port mapping 443:8443"


# ---------------------------------------------------------------------------
# SA-5: compose port-exposure hardening (loopback-by-default binds)
# ---------------------------------------------------------------------------


def _iter_published_ports(
    compose_path: Path,
):
    """Yield (service_name, raw_port_mapping) for every published compose port.

    Reads the raw, un-interpolated ``ports:`` entries so the test asserts on the
    literal ``${HOST_BIND_IP:-127.0.0.1}:`` prefix an operator sees in the file,
    not a value resolved from the current environment. Only short-syntax string
    mappings (``"HOST:CONTAINER"`` / ``"IP:HOST:CONTAINER"``) are yielded; the
    long-syntax dict form is yielded as its raw dict for the caller to inspect.
    """
    data = yaml.safe_load(compose_path.read_text())
    services = data.get("services", {})
    for service_name, service in services.items():
        for entry in service.get("ports", []) or []:
            yield service_name, entry


def _target_port(raw_mapping: str) -> int | None:
    """Extract the container-side target port from a short-syntax mapping.

    Handles ``HOST:CONTAINER`` and ``IP:HOST:CONTAINER`` (the IP segment may
    itself contain a ``${VAR:-default}`` expression, which never contains a
    trailing ``:CONTAINER`` port, so splitting on the final colon is safe).
    Returns None if the container port cannot be parsed as an int.
    """
    container_side = str(raw_mapping).rsplit(":", 1)[-1].strip().strip('"').strip("'")
    # A container port may carry a /protocol suffix (e.g. "53/udp").
    container_side = container_side.split("/", 1)[0]
    try:
        return int(container_side)
    except ValueError:
        return None


@pytest.mark.parametrize("compose_filename", COMPOSE_FILES)
def test_compose_only_front_door_binds_all_interfaces(
    repo_root: Path,
    compose_filename: str,
):
    """Every published port except the nginx front door must be loopback-bound.

    This is the core SA-5 invariant: a fresh checkout must not expose MongoDB,
    the OpenBao vault, IdP admin consoles, or backend MCP servers on 0.0.0.0.
    Only 80->8080 and 443->8443 (the authenticated nginx entry point) may bind
    all interfaces.
    """
    compose_file = repo_root / compose_filename
    assert compose_file.exists(), f"{compose_filename} not found"

    offenders: list[str] = []
    for service_name, entry in _iter_published_ports(compose_file):
        # Long-syntax dict form: assert host_ip is loopback unless front door.
        if isinstance(entry, dict):
            target = entry.get("target")
            if target in FRONT_DOOR_TARGET_PORTS:
                continue
            host_ip = str(entry.get("host_ip", ""))
            if host_ip not in ("127.0.0.1", "::1"):
                offenders.append(f"{service_name}: {entry}")
            continue

        raw = str(entry)
        target = _target_port(raw)
        if target in FRONT_DOOR_TARGET_PORTS:
            # Front door is intentionally published on all interfaces.
            continue
        if not raw.strip().strip('"').strip("'").startswith(LOOPBACK_BIND_PREFIX):
            offenders.append(f"{service_name}: {raw}")

    assert not offenders, (
        f"{compose_filename}: these published ports are not loopback-bound "
        f"(must be prefixed with '{LOOPBACK_BIND_PREFIX}'): {offenders}"
    )


@pytest.mark.parametrize("compose_filename", COMPOSE_FILES)
def test_compose_front_door_still_published(
    repo_root: Path,
    compose_filename: str,
):
    """The nginx front door (80/443) must remain published on all interfaces.

    Guards against an over-eager hardening pass that accidentally loopback-binds
    the public entry point and breaks external access.
    """
    compose_file = repo_root / compose_filename
    content = compose_file.read_text()

    assert "80:8080" in content, f"{compose_filename}: front-door 80:8080 mapping missing"
    assert "443:8443" in content, f"{compose_filename}: front-door 443:8443 mapping missing"
    # The front door must NOT carry a loopback prefix.
    assert f"{LOOPBACK_BIND_PREFIX}80:8080" not in content, (
        f"{compose_filename}: front-door 80:8080 must stay on all interfaces, not loopback"
    )
    assert f"{LOOPBACK_BIND_PREFIX}443:8443" not in content, (
        f"{compose_filename}: front-door 443:8443 must stay on all interfaces, not loopback"
    )


@pytest.mark.parametrize("compose_filename", COMPOSE_FILES)
def test_compose_sensitive_ports_are_loopback(
    repo_root: Path,
    compose_filename: str,
):
    """The highest-risk ports must be loopback-bound wherever they are published.

    MongoDB (27017) and the OpenBao vault (8200) are the two ports whose exposure
    is most damaging (unauthenticated DB access; vault root == all egress
    credentials). This test pins them explicitly so a future edit cannot quietly
    re-expose them even if the generic invariant test is changed.
    """
    compose_file = repo_root / compose_filename
    sensitive_targets = {27017, 8200}

    seen: set[int] = set()
    for _service_name, entry in _iter_published_ports(compose_file):
        if isinstance(entry, dict):
            continue
        raw = str(entry).strip().strip('"').strip("'")
        target = _target_port(raw)
        if target in sensitive_targets:
            seen.add(target)
            assert raw.startswith(LOOPBACK_BIND_PREFIX), (
                f"{compose_filename}: sensitive port {target} must be loopback-bound, got {raw!r}"
            )

    # Not every file publishes both (e.g. some variants omit a service), so we
    # only assert on what is present; nothing to require here beyond the loop.


def test_host_bind_ip_documented_in_env_example(repo_root: Path):
    """HOST_BIND_IP must be documented so operators know how to opt into 0.0.0.0."""
    env_example = repo_root / ".env.example"
    assert env_example.exists(), ".env.example not found"
    content = env_example.read_text()
    assert "HOST_BIND_IP" in content, ".env.example must document HOST_BIND_IP"
