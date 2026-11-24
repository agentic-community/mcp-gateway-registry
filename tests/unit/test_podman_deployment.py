"""Pin the Podman deployment path in build_and_run.sh and docker-compose.podman.yml."""

import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
BUILD_SCRIPT: Path = REPO_ROOT / "build_and_run.sh"
PODMAN_HELPERS: Path = REPO_ROOT / "scripts" / "podman-setup.sh"
PODMAN_COMPOSE: Path = REPO_ROOT / "docker-compose.podman.yml"
MONGO_INIT_SCRIPT: Path = REPO_ROOT / "scripts" / "init-mongodb-ce.py"
SHELL_TIMEOUT_SECONDS: int = 15


def _services() -> dict[str, Any]:
    """Return the services declared in the Podman compose file."""
    return yaml.safe_load(PODMAN_COMPOSE.read_text())["services"]


def _script_lines() -> list[str]:
    """Return build_and_run.sh lines without full-line comments."""
    lines = BUILD_SCRIPT.read_text().splitlines()
    return [line for line in lines if not line.lstrip().startswith("#")]


def _line_index(
    needle: str,
) -> int:
    """Return the first executable script line containing the text."""
    for index, line in enumerate(_script_lines()):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} not found in build_and_run.sh")


def _apply_url_defaults(
    env_path: Path,
) -> None:
    """Run podman_apply_url_defaults against an env file."""
    bash = shutil.which("bash")
    assert bash is not None
    command = f'log() {{ :; }}; . "{PODMAN_HELPERS}"; podman_apply_url_defaults'
    subprocess.run(  # nosec B603 - hardcoded command sourcing a repository script
        [bash, "-c", command],
        cwd=env_path.parent,
        check=True,
        capture_output=True,
        text=True,
        timeout=SHELL_TIMEOUT_SECONDS,
    )


def test_compose_check_uses_selected_runtime() -> None:
    assert "if ! docker compose version" not in BUILD_SCRIPT.read_text()


def test_runtime_check_runs_before_frontend_build() -> None:
    assert _line_index("$COMPOSE_CMD version") < _line_index("npm install")


def test_secrets_exist_before_compose_down() -> None:
    assert _line_index("ensure_secret DOCUMENTDB_PASSWORD") < _line_index("down --remove-orphans")


def test_port_check_runs_before_compose_down() -> None:
    assert _line_index("podman_check_ports") < _line_index("down --remove-orphans")


def test_script_avoids_gnu_only_sed_in_place() -> None:
    assert not [line for line in _script_lines() if "sed -i" in line]


def test_script_does_not_log_metrics_token_prefixes() -> None:
    assert ":0:20" not in BUILD_SCRIPT.read_text()


def test_podman_mongodb_requires_authentication() -> None:
    command = _services()["mongodb"]["command"]
    assert "--auth" in command and "--keyFile" in command


def test_podman_mongodb_healthcheck_has_no_password() -> None:
    healthcheck = " ".join(_services()["mongodb"]["healthcheck"]["test"])
    assert "DOCUMENTDB_PASSWORD" not in healthcheck


def test_podman_init_mounts_every_default_scope_file() -> None:
    expected = set(re.findall(r'"([a-z-]+\.json)"', MONGO_INIT_SCRIPT.read_text()))
    volumes = " ".join(_services()["mongodb-init"]["volumes"])
    assert not [name for name in expected if f"/app/scripts/{name}" not in volumes]


def test_podman_gateway_uses_unprivileged_default_ports() -> None:
    assert _services()["registry"]["ports"] == [
        "${REGISTRY_HTTP_HOST_PORT:-8080}:8080",
        "${REGISTRY_HTTPS_HOST_PORT:-8443}:8443",
    ]


def test_podman_auth_server_port_is_loopback() -> None:
    assert _services()["auth-server"]["ports"] == [
        "${HOST_BIND_IP:-127.0.0.1}:${AUTH_SERVER_HOST_PORT:-8888}:8888",
    ]


def test_auth_server_receives_external_url() -> None:
    environment = _services()["auth-server"]["environment"]
    assert any(item.startswith("AUTH_SERVER_EXTERNAL_URL=") for item in environment)


def test_url_defaults_replace_shipped_docker_values(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AUTH_SERVER_EXTERNAL_URL=http://localhost:8888\n")
    _apply_url_defaults(env_path)
    assert env_path.read_text() == "AUTH_SERVER_EXTERNAL_URL=http://localhost:8080\n"


def test_url_defaults_keep_custom_values(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("REGISTRY_URL=https://registry.example.com\n")
    _apply_url_defaults(env_path)
    assert env_path.read_text() == "REGISTRY_URL=https://registry.example.com\n"


def test_url_defaults_write_private_env_file(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("REGISTRY_URL=http://localhost\n")
    env_path.chmod(0o644)
    _apply_url_defaults(env_path)
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
