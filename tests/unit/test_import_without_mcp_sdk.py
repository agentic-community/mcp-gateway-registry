"""The registry modules the auth server imports must load without the MCP SDK.

The auth-server image installs the ``registry`` package but not the MCP SDK
(``mcp``), which only the registry image needs. A registry module on the auth
server's import path that imports ``mcp`` at module level crashes the auth
server on startup with ``ModuleNotFoundError: No module named 'mcp'``. The test
environment has the SDK installed, so this test blocks it in a subprocess to
reproduce the auth-server image.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Modules on the auth server's import path that pull in registry code. The
# repository factory is the one that failed: it imports the search repository,
# which imports registry/core/mcp_apps.py.
AUTH_SERVER_REGISTRY_IMPORTS: tuple[str, ...] = (
    "registry.core.mcp_apps",
    "registry.repositories.documentdb.search_repository",
    "registry.repositories.factory",
)
IMPORT_TIMEOUT_SECONDS: int = 120
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
# auth_server/server.py imports its siblings by bare name, as it does in the
# image, and refuses to import without its two signing secrets. These values
# are test-only placeholders.
AUTH_SERVER_TEST_ENV: dict[str, str] = {
    "SECRET_KEY": "test-only-not-a-real-secret-0123456789abcdef",
    "AUTH_SERVER_NGINX_MARKER_SECRET": "test-only-not-a-real-secret-0123456789abcdef",
}


def _import_with_mcp_blocked(
    module_name: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Import one module in a fresh interpreter where ``import mcp`` fails."""
    code = f"import sys\nsys.modules['mcp'] = None\nimport {module_name}\n"
    env = {**os.environ, **(extra_env or {})}
    return subprocess.run(  # nosec B603 - fixed interpreter and a hardcoded module name
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=IMPORT_TIMEOUT_SECONDS,
        check=False,
        env=env,
    )


@pytest.mark.unit
@pytest.mark.parametrize("module_name", AUTH_SERVER_REGISTRY_IMPORTS)
def test_module_imports_without_mcp_sdk(module_name):
    """The module loads in an environment without the MCP SDK, as in the auth-server image."""
    result = _import_with_mcp_blocked(module_name)

    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.unit
def test_auth_server_imports_without_mcp_sdk():
    """auth_server.server itself loads without the SDK, as in the auth-server image."""
    pythonpath = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "auth_server")])
    env = {**AUTH_SERVER_TEST_ENV, "PYTHONPATH": pythonpath}

    result = _import_with_mcp_blocked("auth_server.server", env)

    assert result.returncode == 0, result.stderr[-2000:]
