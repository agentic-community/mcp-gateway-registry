"""
Unit tests for Docker container security configuration.

Tests verify that Dockerfiles follow CIS Docker Benchmark 4.1 requirements:
- Non-root USER directive
- No sudo package
- HEALTHCHECK directives
- Proper environment variables (PIP_NO_CACHE_DIR)
"""

import re
from pathlib import Path

import pytest


# List of Dockerfiles to test
DOCKERFILES = [
    "Dockerfile",
    "docker/Dockerfile.auth",
    "docker/Dockerfile.registry",
    "docker/Dockerfile.registry-cpu",
    "docker/Dockerfile.mcp-server",
    "docker/Dockerfile.mcp-server-cpu",
    "docker/Dockerfile.mcp-server-light",
    "docker/Dockerfile.scopes-init",
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
    assert user_pattern.search(
        content
    ), f"{dockerfile_path}: Missing USER directive (CIS 4.1)"


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
    assert (
        last_user.lower() != "root"
    ), f"{dockerfile_path}: Last USER directive is 'root'"


@pytest.mark.parametrize("dockerfile_path", DOCKERFILES)
def test_dockerfile_no_sudo(repo_root: Path, dockerfile_path: str):
    """Test that Dockerfile does not install sudo package."""
    dockerfile = repo_root / dockerfile_path
    assert dockerfile.exists(), f"Dockerfile not found: {dockerfile}"

    content = dockerfile.read_text()

    # Check that sudo is not being installed
    assert (
        "sudo" not in content
    ), f"{dockerfile_path}: Contains 'sudo' package (security risk)"


@pytest.mark.parametrize(
    "dockerfile_path",
    [
        f
        for f in DOCKERFILES
        if "scopes-init" not in f  # Exclude one-shot init containers
    ],
)
def test_dockerfile_has_healthcheck(repo_root: Path, dockerfile_path: str):
    """Test that Dockerfile has HEALTHCHECK directive.

    Note: One-shot init containers (scopes-init) are excluded as they
    don't need health checks - they run once and exit.
    """
    dockerfile = repo_root / dockerfile_path
    assert dockerfile.exists(), f"Dockerfile not found: {dockerfile}"

    content = dockerfile.read_text()

    # Check for HEALTHCHECK directive
    healthcheck_pattern = re.compile(r"^HEALTHCHECK\s+", re.MULTILINE)
    assert healthcheck_pattern.search(
        content
    ), f"{dockerfile_path}: Missing HEALTHCHECK directive"


@pytest.mark.parametrize(
    "dockerfile_path",
    [
        f
        for f in DOCKERFILES
        if not f.startswith("terraform/")  # Exclude Grafana (Node.js)
        and not f.endswith("scopes-init")  # Exclude busybox
        and not f.endswith("metrics-db")  # Exclude alpine-based
    ],
)
def test_python_dockerfile_has_pip_no_cache(
    repo_root: Path, dockerfile_path: str
):
    """Test that Python Dockerfiles set PIP_NO_CACHE_DIR=1."""
    dockerfile = repo_root / dockerfile_path
    assert dockerfile.exists(), f"Dockerfile not found: {dockerfile}"

    content = dockerfile.read_text()

    # Check if it's a Python-based image
    if re.search(r"FROM.*python", content, re.IGNORECASE):
        # Check for PIP_NO_CACHE_DIR
        assert (
            "PIP_NO_CACHE_DIR" in content
        ), f"{dockerfile_path}: Python image missing PIP_NO_CACHE_DIR"


def test_docker_compose_has_security_options(repo_root: Path):
    """Test that docker-compose.yml has security hardening options."""
    compose_file = repo_root / "docker-compose.yml"
    assert compose_file.exists(), "docker-compose.yml not found"

    content = compose_file.read_text()

    # Check for security_opt
    assert (
        "security_opt:" in content
    ), "docker-compose.yml missing security_opt"
    assert (
        "no-new-privileges:true" in content
    ), "docker-compose.yml missing no-new-privileges"

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
        assert "- SETUID" in content, f"{compose_filename}: missing SETUID in cap_add (required by MongoDB gosu)"
        assert "- SETGID" in content, f"{compose_filename}: missing SETGID in cap_add (required by MongoDB gosu)"


def test_docker_compose_registry_port_mapping(repo_root: Path):
    """Test that docker-compose.yml maps nginx to high ports."""
    compose_file = repo_root / "docker-compose.yml"
    assert compose_file.exists(), "docker-compose.yml not found"

    content = compose_file.read_text()

    # Check for port mapping 80:8080 and 443:8443
    assert (
        '"80:8080"' in content or "'80:8080'" in content
    ), "Missing port mapping 80:8080"
    assert (
        '"443:8443"' in content or "'443:8443'" in content
    ), "Missing port mapping 443:8443"
