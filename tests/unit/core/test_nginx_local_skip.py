"""Tests verifying nginx config generation skips local (stdio) servers."""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from registry.constants import HealthStatus
from registry.core.nginx_service import NginxConfigService


@pytest.fixture(autouse=True)
def mock_atomic_write():
    """Stub the config write + validate path so tests don't touch the real
    config path (#1044) and don't shell out to ``nginx -t``.

    ``_write_and_validate_config`` is stubbed to delegate to the (mocked)
    ``_atomic_write_text`` so tests that inspect what was written can declare
    this fixture as a parameter: each recorded call is (path, content), so
    ``mock_atomic_write.call_args_list[i][0][1]`` is the content string.
    """
    with patch("registry.core.nginx_service._atomic_write_text") as m:
        with patch(
            "registry.core.nginx_service._write_and_validate_config",
            side_effect=lambda path, content: m(path, content),
        ):
            yield m


@pytest.fixture
def nginx_service():
    with patch("registry.core.nginx_service.Path") as mock_path_class:
        mock_template = MagicMock()
        mock_template.exists.return_value = True
        mock_path_class.return_value = mock_template
        with patch("registry.core.nginx_service.settings") as mock_settings:
            mock_settings.nginx_updates_enabled = True
            mock_settings.deployment_mode = MagicMock()
            mock_settings.deployment_mode.value = "with-gateway"
            mock_settings.nginx_config_path = "/tmp/nginx.conf"
            yield NginxConfigService()


@pytest.fixture
def mock_health_service():
    s = MagicMock()
    s.server_health_status = {}
    return s


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_server_excluded_from_location_blocks(
    nginx_service, mock_health_service, mock_atomic_write
):
    """Local servers must not produce a proxy_pass location block."""
    template_content = "server { {{LOCATION_BLOCKS}} }"
    servers = {
        "/remote": {
            "server_name": "remote",
            "proxy_pass_url": "http://upstream/mcp",
            "supported_transports": ["streamable-http"],
            "deployment": "remote",
        },
        "/local": {
            "server_name": "local",
            "deployment": "local",
            "local_runtime": {"type": "npx", "package": "@acme/mcp"},
            "supported_transports": ["stdio"],
        },
    }
    mock_health_service.server_health_status = {
        "/remote": HealthStatus.HEALTHY,
        "/local": HealthStatus.LOCAL,
    }

    with patch("builtins.open", mock_open(read_data=template_content)):
        with patch("registry.health.service.health_service", mock_health_service):
            with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                with patch.object(nginx_service, "reload_nginx", return_value=True):
                    with patch("os.environ.get", return_value="http://keycloak:8080"):
                        await nginx_service.generate_config_async(servers)

    # Inspect what was passed to _atomic_write_text
    rendered = "\n".join(call.args[1] for call in mock_atomic_write.call_args_list)
    # Remote upstream should appear in the config; local server's path must NOT.
    assert "http://upstream" in rendered or "/remote" in rendered
    # Critically, the local server should not produce a proxy_pass block keyed on its path
    assert "location /local/" not in rendered
    # And no commented "service currently unhealthy" stub for the local server
    assert "/local/" not in rendered or "deployment" not in rendered
