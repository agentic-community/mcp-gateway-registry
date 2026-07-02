"""
Unit tests for registry/core/nginx_service.py

Tests the NginxConfigService for configuration generation and reload.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, mock_open, patch
from urllib.parse import urlparse

import httpx
import pytest

from registry.constants import HealthStatus
from registry.core import nginx_service as nginx_module
from registry.core.nginx_service import NginxConfigService

# Capture the real write-validate-promote helpers before the autouse fixture
# stubs the module attributes, so the dedicated validation tests can exercise
# the genuine flow (temp write, nginx -t gate, promote-or-restore).
_REAL_WRITE_AND_VALIDATE = nginx_module._write_and_validate_config
_REAL_ATOMIC_WRITE = nginx_module._atomic_write_text

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def mock_atomic_write():
    """Stub the config write + validate path so tests don't touch /etc/nginx.

    The real helpers do tempfile + os.replace against
    settings.nginx_config_path and run ``nginx -t``. In unit tests the path is a
    fixture string ('/etc/nginx/conf.d/nginx_rev_proxy.conf') that the test user
    typically can't write to, and no nginx binary is present. Patching here keeps
    every test that reaches generate_config_async hermetic.

    ``_write_and_validate_config`` is stubbed to delegate to the (mocked)
    ``_atomic_write_text`` so tests that inspect what was written can declare
    this fixture as a parameter and read ``mock_atomic_write.call_args_list`` -
    each call is (path, content). The dedicated validation tests below opt out
    of this stub to exercise the real write-validate-promote flow.
    """
    with patch("registry.core.nginx_service._atomic_write_text") as mock_write:
        with patch(
            "registry.core.nginx_service._write_and_validate_config",
            side_effect=lambda path, content: mock_write(path, content),
        ):
            yield mock_write


@pytest.fixture
def nginx_service():
    """Create a NginxConfigService instance."""
    with patch("registry.core.nginx_service.Path") as mock_path_class:
        # Mock SSL certificate existence checks
        mock_ssl_cert = MagicMock()
        mock_ssl_cert.exists.return_value = False
        mock_ssl_key = MagicMock()
        mock_ssl_key.exists.return_value = False

        # Mock template path existence
        mock_template = MagicMock()
        mock_template.exists.return_value = True

        mock_path_class.return_value = mock_template

        # Mock settings.nginx_updates_enabled to True for testing
        with patch("registry.core.nginx_service.settings") as mock_settings:
            mock_settings.nginx_updates_enabled = True
            mock_settings.deployment_mode = MagicMock()
            mock_settings.deployment_mode.value = "with-gateway"
            mock_settings.nginx_config_path = "/etc/nginx/conf.d/nginx_rev_proxy.conf"
            mock_settings.auth_server_url = "http://auth-server:8888"

            service = NginxConfigService()
            yield service


@pytest.fixture
def sample_servers():
    """Create sample server configuration."""
    return {
        "/test-server": {
            "server_name": "test-server",
            "proxy_pass_url": "http://localhost:8000/mcp",
            "supported_transports": ["streamable-http"],
            "headers": [{"X-Custom-Header": "value"}],
        },
        "/test-server-2": {
            "server_name": "test-server-2",
            "proxy_pass_url": "https://external.example.com/sse",
            "supported_transports": ["sse"],
        },
    }


@pytest.fixture
def mock_health_service():
    """Create mock health service."""
    mock_service = MagicMock()
    mock_service.server_health_status = {}
    return mock_service


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


@pytest.mark.unit
def test_nginx_service_init_http_only():
    """Test NginxConfigService initialization with HTTP-only template."""
    with patch("registry.core.nginx_service.Path") as mock_path_class:
        # Mock SSL certificates as not existing
        mock_ssl_cert = MagicMock()
        mock_ssl_cert.exists.return_value = False
        mock_ssl_key = MagicMock()
        mock_ssl_key.exists.return_value = False

        # Mock template paths - return Path-like mocks that stringify correctly
        mock_http_only_template = MagicMock()
        mock_http_only_template.exists.return_value = True
        mock_http_only_template.__str__ = MagicMock(return_value="/templates/nginx_http_only.conf")

        def path_side_effect(path_str):
            if "fullchain.pem" in str(path_str):
                return mock_ssl_cert
            elif "privkey.pem" in str(path_str):
                return mock_ssl_key
            elif "http_only" in str(path_str).lower():
                return mock_http_only_template
            else:
                # For any other path (like http_and_https), return non-existent
                mock = MagicMock()
                mock.exists.return_value = False
                return mock

        mock_path_class.side_effect = path_side_effect

        service = NginxConfigService()

        # Should use HTTP-only template
        assert "http_only" in str(service.nginx_template_path).lower()


@pytest.mark.unit
def test_nginx_service_init_http_and_https():
    """Test NginxConfigService initialization with HTTPS template."""
    with patch("registry.core.nginx_service.Path") as mock_path_class:
        # Mock SSL certificates as existing
        mock_ssl_cert = MagicMock()
        mock_ssl_cert.exists.return_value = True
        mock_ssl_key = MagicMock()
        mock_ssl_key.exists.return_value = True

        # Mock template path with proper string representation
        mock_https_template = MagicMock()
        mock_https_template.exists.return_value = True
        mock_https_template.__str__ = MagicMock(return_value="/templates/nginx_http_and_https.conf")

        def path_side_effect(path_str):
            if "fullchain.pem" in str(path_str):
                return mock_ssl_cert
            elif "privkey.pem" in str(path_str):
                return mock_ssl_key
            elif "http_and_https" in str(path_str).lower():
                return mock_https_template
            else:
                mock = MagicMock()
                mock.exists.return_value = False
                return mock

        mock_path_class.side_effect = path_side_effect

        service = NginxConfigService()

        # Should use HTTP+HTTPS template
        assert "http_and_https" in str(service.nginx_template_path).lower()


# =============================================================================
# GET_ADDITIONAL_SERVER_NAMES TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_additional_server_names_from_env(nginx_service):
    """Test getting additional server names from environment variable."""
    with patch.dict("os.environ", {"GATEWAY_ADDITIONAL_SERVER_NAMES": "custom.example.com"}):
        result = await nginx_service.get_additional_server_names()

        assert result == "custom.example.com"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_additional_server_names_ec2_metadata(nginx_service):
    """Test getting additional server names from EC2 metadata."""
    with patch.dict("os.environ", {}, clear=True):
        mock_client = AsyncMock()

        # Mock token response
        mock_token_response = MagicMock()
        mock_token_response.status_code = 200
        mock_token_response.text = "test-token"

        # Mock IP response
        mock_ip_response = MagicMock()
        mock_ip_response.status_code = 200
        mock_ip_response.text = "10.0.1.100"

        mock_client.put.return_value = mock_token_response
        mock_client.get.return_value = mock_ip_response

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await nginx_service.get_additional_server_names()

            assert result == "10.0.1.100"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_additional_server_names_ecs_metadata(nginx_service):
    """Test getting additional server names from ECS metadata."""

    with patch.dict("os.environ", {"ECS_CONTAINER_METADATA_URI": "http://169.254.170.2/v4/test"}):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"Networks": [{"IPv4Addresses": ["172.17.0.5"]}]}'

        mock_client.get.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await nginx_service.get_additional_server_names()

            assert result == "172.17.0.5"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_additional_server_names_pod_ip(nginx_service):
    """Test getting additional server names from Kubernetes POD_IP."""
    # Mock httpx to fail (simulating no EC2/ECS metadata available)
    mock_client = AsyncMock()
    mock_client.put.side_effect = httpx.ConnectTimeout("Connection timed out")
    mock_client.get.side_effect = httpx.ConnectTimeout("Connection timed out")

    with patch.dict("os.environ", {"POD_IP": "192.168.1.50"}, clear=False):
        # Clear metadata-related env vars
        with patch.dict("os.environ", {"ECS_CONTAINER_METADATA_URI": ""}, clear=False):
            with patch("httpx.AsyncClient") as mock_async_client:
                mock_async_client.return_value.__aenter__.return_value = mock_client

                result = await nginx_service.get_additional_server_names()

                assert result == "192.168.1.50"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_additional_server_names_hostname_command(nginx_service):
    """Test getting additional server names from hostname command."""
    with patch.dict("os.environ", {}, clear=True):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "10.1.1.1 192.168.1.1 "

        with patch("subprocess.run", return_value=mock_result):
            with patch("httpx.AsyncClient") as mock_client:
                # Mock EC2 metadata failure
                mock_client.return_value.__aenter__.return_value.put.side_effect = (
                    httpx.ConnectError("No connection")
                )

                result = await nginx_service.get_additional_server_names()

                assert result == "10.1.1.1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_additional_server_names_fallback_empty(nginx_service):
    """Test getting additional server names with no available sources."""
    with patch.dict("os.environ", {}, clear=True):
        with patch("httpx.AsyncClient") as mock_client:
            # Mock EC2 metadata failure
            mock_client.return_value.__aenter__.return_value.put.side_effect = httpx.ConnectError(
                "No connection"
            )

            with patch("subprocess.run") as mock_subprocess:
                # Mock hostname command failure
                mock_subprocess.side_effect = Exception("Command failed")

                result = await nginx_service.get_additional_server_names()

                assert result == ""


# =============================================================================
# GENERATE_CONFIG TESTS
# =============================================================================


@pytest.mark.unit
def test_generate_config_from_async_context(nginx_service):
    """Test that generate_config logs error when called from async context."""

    async def async_test():
        result = nginx_service.generate_config({})
        assert result is False

    asyncio.run(async_test())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_success(nginx_service, sample_servers, mock_health_service):
    """Test successful configuration generation."""
    template_content = """
server {
    listen 80;
    server_name localhost {{ADDITIONAL_SERVER_NAMES}};

{{LOCATION_BLOCKS}}
}
"""

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)):
            with patch("registry.health.service.health_service", mock_health_service):
                # Mark servers as healthy
                mock_health_service.server_health_status = {
                    "/test-server": HealthStatus.HEALTHY,
                    "/test-server-2": HealthStatus.HEALTHY,
                }

                with patch.object(
                    nginx_service, "get_additional_server_names", return_value="10.0.0.1"
                ):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        env_values = {
                            "AUTH_PROVIDER": "keycloak",
                            "KEYCLOAK_URL": "http://keycloak:8080",
                            "NGINX_DISABLE_API_AUTH_REQUEST": "false",
                        }
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            result = await nginx_service.generate_config_async(sample_servers)

                            assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_template_not_found(nginx_service, sample_servers):
    """Test configuration generation when template is not found."""
    with patch.object(nginx_service.nginx_template_path, "exists", return_value=False):
        result = await nginx_service.generate_config_async(sample_servers)

        assert result is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_unhealthy_servers(
    nginx_service, sample_servers, mock_health_service
):
    """Test configuration generation with unhealthy servers."""
    template_content = """
server {
    listen 80;
{{LOCATION_BLOCKS}}
}
"""

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)) as mock_file:
            with patch("registry.health.service.health_service", mock_health_service):
                # Mark servers as unhealthy
                mock_health_service.server_health_status = {
                    "/test-server": HealthStatus.UNHEALTHY_TIMEOUT,
                    "/test-server-2": HealthStatus.UNHEALTHY_CONNECTION_ERROR,
                }

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        with patch("os.environ.get", return_value="http://keycloak:8080"):
                            result = await nginx_service.generate_config_async(sample_servers)

                            assert result is True

                            # Verify that config was written
                            mock_file.assert_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_exception(nginx_service, sample_servers):
    """Test configuration generation with exception."""
    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", side_effect=Exception("File error")):
            result = await nginx_service.generate_config_async(sample_servers)

            assert result is False


# =============================================================================
# RELOAD_NGINX TESTS
# =============================================================================


@pytest.mark.unit
def test_reload_nginx_success(nginx_service):
    """Test successful Nginx reload."""
    mock_test_result = MagicMock()
    mock_test_result.returncode = 0

    mock_reload_result = MagicMock()
    mock_reload_result.returncode = 0

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [mock_test_result, mock_reload_result]

        result = nginx_service.reload_nginx()

        assert result is True
        assert mock_run.call_count == 2


@pytest.mark.unit
def test_reload_nginx_config_test_failure(nginx_service):
    """Test Nginx reload when config test fails."""
    mock_test_result = MagicMock()
    mock_test_result.returncode = 1
    mock_test_result.stderr = "Config error"

    with patch("subprocess.run", return_value=mock_test_result):
        result = nginx_service.reload_nginx()

        assert result is False


@pytest.mark.unit
def test_reload_nginx_reload_failure(nginx_service):
    """Test Nginx reload when reload command fails."""
    mock_test_result = MagicMock()
    mock_test_result.returncode = 0

    mock_reload_result = MagicMock()
    mock_reload_result.returncode = 1
    mock_reload_result.stderr = "Reload failed"

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [mock_test_result, mock_reload_result]

        result = nginx_service.reload_nginx()

        assert result is False


@pytest.mark.unit
def test_reload_nginx_not_found(nginx_service):
    """Test Nginx reload when nginx command is not found."""
    with patch("subprocess.run", side_effect=FileNotFoundError("nginx not found")):
        result = nginx_service.reload_nginx()

        assert result is False


@pytest.mark.unit
def test_reload_nginx_exception(nginx_service):
    """Test Nginx reload with unexpected exception."""
    with patch("subprocess.run", side_effect=Exception("Unexpected error")):
        result = nginx_service.reload_nginx()

        assert result is False


# =============================================================================
# TRANSPORT LOCATION BLOCKS TESTS
# =============================================================================


@pytest.mark.unit
def test_generate_transport_location_blocks_streamable_http(nginx_service):
    """Test generating location blocks for streamable-http transport."""
    server_info = {
        "proxy_pass_url": "http://localhost:8000/mcp",
        "supported_transports": ["streamable-http"],
    }

    blocks = nginx_service._generate_transport_location_blocks("/test", server_info)

    assert len(blocks) == 1
    assert "location {{ROOT_PATH}}/test" in blocks[0]
    # Issue #1026 - MCP traffic is routed through auth_server mcp-proxy.
    assert "proxy_pass http://auth-server:8888/mcp-proxy/test/" in blocks[0]
    # $backend_url is set in the rewrite phase so the /validate
    # subrequest can bind it into the internal token, then forwarded as X-Upstream-Url.
    assert 'set $backend_url "http://localhost:8000/mcp"' in blocks[0]
    assert "proxy_set_header X-Upstream-Url $backend_url" in blocks[0]


@pytest.mark.unit
def test_generate_transport_location_blocks_sse(nginx_service):
    """Test generating location blocks for SSE transport."""
    server_info = {
        "proxy_pass_url": "http://localhost:8000/sse",
        "supported_transports": ["sse"],
    }

    blocks = nginx_service._generate_transport_location_blocks("/test", server_info)

    assert len(blocks) == 1
    assert "location {{ROOT_PATH}}/test" in blocks[0]
    # Issue #1026 - MCP traffic is routed through auth_server mcp-proxy.
    assert "proxy_pass http://auth-server:8888/mcp-proxy/test/" in blocks[0]
    assert 'set $backend_url "http://localhost:8000/sse"' in blocks[0]
    assert "proxy_set_header X-Upstream-Url $backend_url" in blocks[0]


@pytest.mark.unit
def test_generate_transport_location_blocks_both_transports(nginx_service):
    """Test generating location blocks when both transports are supported."""
    server_info = {
        "proxy_pass_url": "http://localhost:8000/mcp",
        "supported_transports": ["streamable-http", "sse"],
    }

    blocks = nginx_service._generate_transport_location_blocks("/test", server_info)

    # Should prefer streamable-http
    assert len(blocks) == 1
    assert "location {{ROOT_PATH}}/test" in blocks[0]


@pytest.mark.unit
def test_generate_transport_location_blocks_no_transports(nginx_service):
    """Test generating location blocks with no specified transports."""
    server_info = {
        "proxy_pass_url": "http://localhost:8000",
        "supported_transports": [],
    }

    blocks = nginx_service._generate_transport_location_blocks("/test", server_info)

    # Should default to streamable-http
    assert len(blocks) == 1
    assert "location {{ROOT_PATH}}/test" in blocks[0]


# =============================================================================
# CREATE_LOCATION_BLOCK TESTS
# =============================================================================


@pytest.mark.unit
def test_create_location_block_streamable_http(nginx_service):
    """Test creating location block for streamable-http."""
    block = nginx_service._create_location_block(
        "/test", "http://localhost:8000/mcp", "streamable-http"
    )

    assert "location {{ROOT_PATH}}/test" in block
    # Issue #1026 - proxy hop lands on auth_server mcp-proxy, upstream goes in header.
    assert "proxy_pass http://auth-server:8888/mcp-proxy/test/" in block
    assert 'set $backend_url "http://localhost:8000/mcp"' in block
    assert "proxy_set_header X-Upstream-Url $backend_url" in block
    # capture + forward the /validate-minted internal token.
    assert "auth_request_set $auth_internal_token $upstream_http_x_internal_token" in block
    assert "proxy_set_header X-Internal-Token $auth_internal_token" in block
    assert "proxy_buffering off" in block
    assert "auth_request /validate" in block
    # Upstream timeouts derived from MCP_PROXY_TIMEOUT so long-running MCP tool
    # calls aren't severed by nginx before the inner auth-server hop times out.
    # The exact read/send value is asserted in TestResolveMcpProxyReadTimeout;
    # here we only assert the directives are emitted (no unresolved f-string).
    assert "proxy_read_timeout " in block
    assert "proxy_send_timeout " in block
    assert "proxy_connect_timeout 10s" in block
    assert "{mcp_proxy_read_timeout}" not in block


@pytest.mark.unit
def test_create_location_block_sse(nginx_service):
    """Test creating location block for SSE."""
    block = nginx_service._create_location_block("/test", "http://localhost:8000/sse", "sse")

    assert "location {{ROOT_PATH}}/test" in block
    # Issue #1026 - proxy hop lands on auth_server mcp-proxy, upstream goes in header.
    assert "proxy_pass http://auth-server:8888/mcp-proxy/test/" in block
    assert 'set $backend_url "http://localhost:8000/sse"' in block
    assert "proxy_set_header X-Upstream-Url $backend_url" in block
    assert "proxy_buffering off" in block
    assert "proxy_set_header Connection $http_connection" in block


@pytest.mark.unit
def test_create_location_block_external_service(nginx_service):
    """Test creating location block for external HTTPS service."""
    block = nginx_service._create_location_block(
        "/test", "https://api.example.com/mcp", "streamable-http"
    )

    assert "location {{ROOT_PATH}}/test" in block
    # Issue #1026 - proxy hop lands on auth_server mcp-proxy, upstream goes in header.
    assert "proxy_pass http://auth-server:8888/mcp-proxy/test/" in block
    assert 'set $backend_url "https://api.example.com/mcp"' in block
    assert "proxy_set_header X-Upstream-Url $backend_url" in block
    # Should use upstream hostname for external services
    assert "proxy_set_header Host api.example.com" in block


@pytest.mark.unit
def test_create_location_block_internal_service(nginx_service):
    """Test creating location block for internal service."""
    block = nginx_service._create_location_block(
        "/test", "http://backend:8000/mcp", "streamable-http"
    )

    assert "location {{ROOT_PATH}}/test" in block
    # Issue #1026 - proxy hop lands on auth_server mcp-proxy, upstream goes in header.
    assert "proxy_pass http://auth-server:8888/mcp-proxy/test/" in block
    assert 'set $backend_url "http://backend:8000/mcp"' in block
    assert "proxy_set_header X-Upstream-Url $backend_url" in block
    # Should preserve original host for internal services
    assert "proxy_set_header Host $host" in block


@pytest.mark.unit
def test_create_location_block_direct_transport(nginx_service):
    """Test creating location block for direct transport."""
    block = nginx_service._create_location_block("/test", "http://localhost:8000", "direct")

    assert "location {{ROOT_PATH}}/test" in block
    # Issue #1026 - proxy hop lands on auth_server mcp-proxy, upstream goes in header.
    assert "proxy_pass http://auth-server:8888/mcp-proxy/test/" in block
    assert 'set $backend_url "http://localhost:8000"' in block
    assert "proxy_set_header X-Upstream-Url $backend_url" in block
    assert "proxy_cache off" in block


# =============================================================================
# KEYCLOAK CONFIGURATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_keycloak_parsing(
    nginx_service, sample_servers, mock_health_service, mock_atomic_write
):
    """Test Keycloak URL parsing in configuration generation."""
    template_content = """
server {
    proxy_pass {{KEYCLOAK_SCHEME}}://{{KEYCLOAK_HOST}}:{{KEYCLOAK_PORT}};
{{LOCATION_BLOCKS}}
}
"""

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)) as mock_file:
            with patch("registry.health.service.health_service", mock_health_service):
                mock_health_service.server_health_status = {
                    "/test-server": HealthStatus.HEALTHY,
                }

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        env_values = {
                            "AUTH_PROVIDER": "keycloak",
                            "KEYCLOAK_URL": "https://keycloak.example.com:8443",
                            "NGINX_DISABLE_API_AUTH_REQUEST": "false",
                        }
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            result = await nginx_service.generate_config_async(sample_servers)

                            assert result is True

                            # Verify file was written with parsed Keycloak values
                            write_calls = list(mock_atomic_write.call_args_list)
                            assert len(write_calls) > 0
                            written_content = write_calls[0][0][1]
                            # Verify the template variables were substituted with
                            # the parsed Keycloak URL components
                            parsed_keycloak = urlparse("https://keycloak.example.com:8443")
                            assert parsed_keycloak.hostname in written_content
                            assert str(parsed_keycloak.port) in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_keycloak_default_port(
    nginx_service, sample_servers, mock_health_service
):
    """Test Keycloak URL parsing with default port."""
    template_content = """
server {
{{KEYCLOAK_SCHEME}}://{{KEYCLOAK_HOST}}:{{KEYCLOAK_PORT}}
{{LOCATION_BLOCKS}}
}
"""

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)):
            with patch("registry.health.service.health_service", mock_health_service):
                mock_health_service.server_health_status = {}

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        env_values = {
                            "AUTH_PROVIDER": "keycloak",
                            "KEYCLOAK_URL": "http://keycloak",
                            "NGINX_DISABLE_API_AUTH_REQUEST": "false",
                        }
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            result = await nginx_service.generate_config_async(sample_servers)

                            assert result is True


# =============================================================================
# KEYCLOAK CONDITIONAL LOCATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_strips_keycloak_locations_for_entra(
    nginx_service, sample_servers, mock_health_service, mock_atomic_write
):
    """Test that Keycloak location blocks are stripped when AUTH_PROVIDER is entra."""
    template_content = """
server {
    listen 80;
    server_name localhost {{ADDITIONAL_SERVER_NAMES}};

    # {{KEYCLOAK_LOCATIONS_START}}
    location /keycloak/ {
        proxy_pass {{KEYCLOAK_SCHEME}}://{{KEYCLOAK_HOST}}:{{KEYCLOAK_PORT}}/;
    }

    location /realms/ {
        proxy_pass {{KEYCLOAK_SCHEME}}://{{KEYCLOAK_HOST}}:{{KEYCLOAK_PORT}}/realms/;
    }
    # {{KEYCLOAK_LOCATIONS_END}}

{{LOCATION_BLOCKS}}
}
"""

    env_values = {
        "AUTH_PROVIDER": "entra",
        "KEYCLOAK_URL": "http://keycloak:8080",
        "NGINX_DISABLE_API_AUTH_REQUEST": "false",
    }

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)) as mock_file:
            with patch("registry.health.service.health_service", mock_health_service):
                mock_health_service.server_health_status = {
                    "/test-server": HealthStatus.HEALTHY,
                }

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            result = await nginx_service.generate_config_async(sample_servers)

                            assert result is True

                            # Verify the written config does not contain keycloak locations
                            write_calls = mock_atomic_write.call_args_list
                            assert len(write_calls) > 0
                            written_content = write_calls[0][0][1]
                            assert "/keycloak/" not in written_content
                            assert "/realms/" not in written_content
                            assert "KEYCLOAK_LOCATIONS_START" not in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_keeps_keycloak_locations_for_keycloak(
    nginx_service, sample_servers, mock_health_service, mock_atomic_write
):
    """Test that Keycloak location blocks are kept when AUTH_PROVIDER is keycloak."""
    template_content = """
server {
    listen 80;
    server_name localhost {{ADDITIONAL_SERVER_NAMES}};

    # {{KEYCLOAK_LOCATIONS_START}}
    location /keycloak/ {
        proxy_pass {{KEYCLOAK_SCHEME}}://{{KEYCLOAK_HOST}}:{{KEYCLOAK_PORT}}/;
    }

    location /realms/ {
        proxy_pass {{KEYCLOAK_SCHEME}}://{{KEYCLOAK_HOST}}:{{KEYCLOAK_PORT}}/realms/;
    }
    # {{KEYCLOAK_LOCATIONS_END}}

{{LOCATION_BLOCKS}}
}
"""

    env_values = {
        "AUTH_PROVIDER": "keycloak",
        "KEYCLOAK_URL": "https://keycloak.example.com:8443",
        "NGINX_DISABLE_API_AUTH_REQUEST": "false",
    }

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)) as mock_file:
            with patch("registry.health.service.health_service", mock_health_service):
                mock_health_service.server_health_status = {
                    "/test-server": HealthStatus.HEALTHY,
                }

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            result = await nginx_service.generate_config_async(sample_servers)

                            assert result is True

                            # Verify the written config contains keycloak locations with substituted values
                            write_calls = mock_atomic_write.call_args_list
                            assert len(write_calls) > 0
                            written_content = write_calls[0][0][1]
                            assert "/keycloak/" in written_content
                            assert "/realms/" in written_content
                            # Verify the template variables were substituted with
                            # the parsed Keycloak URL components
                            parsed_keycloak = urlparse("https://keycloak.example.com:8443")
                            assert parsed_keycloak.hostname in written_content
                            assert str(parsed_keycloak.port) in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_strips_keycloak_locations_for_cognito(
    nginx_service, sample_servers, mock_health_service, mock_atomic_write
):
    """Test that Keycloak location blocks are stripped when AUTH_PROVIDER is cognito."""
    template_content = """
server {
    listen 80;
    server_name localhost {{ADDITIONAL_SERVER_NAMES}};

    # {{KEYCLOAK_LOCATIONS_START}}
    location /keycloak/ {
        proxy_pass {{KEYCLOAK_SCHEME}}://{{KEYCLOAK_HOST}}:{{KEYCLOAK_PORT}}/;
    }
    # {{KEYCLOAK_LOCATIONS_END}}

{{LOCATION_BLOCKS}}
}
"""

    env_values = {
        "AUTH_PROVIDER": "cognito",
        "NGINX_DISABLE_API_AUTH_REQUEST": "false",
    }

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)) as mock_file:
            with patch("registry.health.service.health_service", mock_health_service):
                mock_health_service.server_health_status = {}

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            result = await nginx_service.generate_config_async(sample_servers)

                            assert result is True

                            write_calls = mock_atomic_write.call_args_list
                            assert len(write_calls) > 0
                            written_content = write_calls[0][0][1]
                            assert "/keycloak/" not in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_keycloak_https_default_port(
    nginx_service, sample_servers, mock_health_service, mock_atomic_write
):
    """Test Keycloak URL parsing defaults to port 443 for HTTPS without explicit port."""
    template_content = """
server {
    {{KEYCLOAK_SCHEME}}://{{KEYCLOAK_HOST}}:{{KEYCLOAK_PORT}}
    {{LOCATION_BLOCKS}}
}
"""

    env_values = {
        "AUTH_PROVIDER": "keycloak",
        "KEYCLOAK_URL": "https://keycloak.example.com",
        "NGINX_DISABLE_API_AUTH_REQUEST": "false",
    }

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)) as mock_file:
            with patch("registry.health.service.health_service", mock_health_service):
                mock_health_service.server_health_status = {}

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            result = await nginx_service.generate_config_async(sample_servers)

                            assert result is True
                            written_content = mock_atomic_write.call_args_list[0][0][1]
                            assert "https" in written_content
                            assert "443" in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_keycloak_hostname_fallback(
    nginx_service, sample_servers, mock_health_service, mock_atomic_write
):
    """Test Keycloak hostname fallback when hostname resolves to bare 'keycloak'."""
    template_content = """
server {
    {{KEYCLOAK_SCHEME}}://{{KEYCLOAK_HOST}}:{{KEYCLOAK_PORT}}
    {{LOCATION_BLOCKS}}
}
"""

    env_values = {
        "AUTH_PROVIDER": "keycloak",
        "KEYCLOAK_URL": "http://keycloak:8080",
        "NGINX_DISABLE_API_AUTH_REQUEST": "false",
    }

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)) as mock_file:
            with patch("registry.health.service.health_service", mock_health_service):
                mock_health_service.server_health_status = {}

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            result = await nginx_service.generate_config_async(sample_servers)

                            assert result is True
                            written_content = mock_atomic_write.call_args_list[0][0][1]
                            # Should still contain keycloak as the host (netloc fallback)
                            assert "keycloak" in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_keycloak_url_parse_exception(
    nginx_service, sample_servers, mock_health_service, mock_atomic_write
):
    """Test Keycloak URL parsing falls back to defaults on exception."""
    template_content = """
server {
    {{KEYCLOAK_SCHEME}}://{{KEYCLOAK_HOST}}:{{KEYCLOAK_PORT}}
    {{LOCATION_BLOCKS}}
}
"""

    env_values = {
        "AUTH_PROVIDER": "keycloak",
        "KEYCLOAK_URL": "http://keycloak:8080",
        "NGINX_DISABLE_API_AUTH_REQUEST": "false",
    }

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)) as mock_file:
            with patch("registry.health.service.health_service", mock_health_service):
                mock_health_service.server_health_status = {}

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            # Force urlparse to raise an exception
                            with patch(
                                "registry.core.nginx_service.urlparse",
                                side_effect=Exception("parse error"),
                            ):
                                result = await nginx_service.generate_config_async(sample_servers)

                                assert result is True
                                written_content = mock_atomic_write.call_args_list[0][0][1]
                                # Should fall back to defaults
                                assert "http" in written_content
                                assert "keycloak" in written_content
                                assert "8080" in written_content


# =============================================================================
# AUTH_SERVER_URL PLACEHOLDER SUBSTITUTION (#553)
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_auth_server_url_parsing(
    nginx_service, sample_servers, mock_health_service, mock_atomic_write
):
    """Test AUTH_SERVER_URL parsing substitutes placeholders (#553)."""
    template_content = """
server {
    proxy_pass http://{{AUTH_SERVER_HOST}}:{{AUTH_SERVER_PORT}}/validate;
{{LOCATION_BLOCKS}}
}
"""

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)):
            with patch("registry.health.service.health_service", mock_health_service):
                mock_health_service.server_health_status = {
                    "/test-server": HealthStatus.HEALTHY,
                }

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        env_values = {
                            "AUTH_PROVIDER": "keycloak",
                            "KEYCLOAK_URL": "http://keycloak:8080",
                            "AUTH_SERVER_URL": "http://auth.internal.svc.cluster.local:8888",
                            "NGINX_DISABLE_API_AUTH_REQUEST": "false",
                        }
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            result = await nginx_service.generate_config_async(sample_servers)

                            assert result is True

                            written_content = mock_atomic_write.call_args_list[0][0][1]
                            assert "auth.internal.svc.cluster.local" in written_content
                            assert "8888" in written_content
                            assert "{{AUTH_SERVER_HOST}}" not in written_content
                            assert "{{AUTH_SERVER_PORT}}" not in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_auth_server_url_default(
    nginx_service, sample_servers, mock_health_service, mock_atomic_write
):
    """Test AUTH_SERVER_URL defaults to auth-server:8888 when unset (#553)."""
    template_content = """
server {
    proxy_pass http://{{AUTH_SERVER_HOST}}:{{AUTH_SERVER_PORT}}/validate;
{{LOCATION_BLOCKS}}
}
"""

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)):
            with patch("registry.health.service.health_service", mock_health_service):
                mock_health_service.server_health_status = {}

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        env_values = {
                            "AUTH_PROVIDER": "keycloak",
                            "KEYCLOAK_URL": "http://keycloak:8080",
                            "NGINX_DISABLE_API_AUTH_REQUEST": "false",
                        }
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            result = await nginx_service.generate_config_async(sample_servers)

                            assert result is True
                            written_content = mock_atomic_write.call_args_list[0][0][1]
                            assert "auth-server" in written_content
                            assert "8888" in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_auth_server_url_parse_fallback(
    nginx_service, sample_servers, mock_health_service, mock_atomic_write
):
    """Test AUTH_SERVER_URL falls back to defaults on parse exception (#553)."""
    template_content = """
server {
    proxy_pass http://{{AUTH_SERVER_HOST}}:{{AUTH_SERVER_PORT}}/validate;
{{LOCATION_BLOCKS}}
}
"""

    env_values = {
        "AUTH_PROVIDER": "keycloak",
        "KEYCLOAK_URL": "http://keycloak:8080",
        "AUTH_SERVER_URL": "http://auth.example.com:9999",
        "NGINX_DISABLE_API_AUTH_REQUEST": "false",
    }

    with patch.object(nginx_service.nginx_template_path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=template_content)):
            with patch("registry.health.service.health_service", mock_health_service):
                mock_health_service.server_health_status = {}

                with patch.object(nginx_service, "get_additional_server_names", return_value=""):
                    with patch.object(nginx_service, "reload_nginx", return_value=True):
                        with patch(
                            "os.environ.get",
                            side_effect=lambda key, default=None: env_values.get(key, default),
                        ):
                            # Force urlparse to raise when parsing AUTH_SERVER_URL.
                            # We need a targeted side_effect: let Keycloak/PF parses
                            # succeed but fail the auth-server parse.
                            original_urlparse = urlparse
                            call_count = {"n": 0}

                            def _selective_urlparse(url):
                                call_count["n"] += 1
                                # The auth-server parse is the 3rd urlparse call
                                # (after Keycloak and PingFederate).
                                if "auth.example.com" in url:
                                    raise Exception("parse error")
                                return original_urlparse(url)

                            with patch(
                                "registry.core.nginx_service.urlparse",
                                side_effect=_selective_urlparse,
                            ):
                                result = await nginx_service.generate_config_async(sample_servers)

                                assert result is True
                                written_content = mock_atomic_write.call_args_list[0][0][1]
                                # Should fall back to defaults
                                assert "auth-server" in written_content
                                assert "8888" in written_content


# =============================================================================
# server-scope $backend_url default must NOT exist in conf templates
# =============================================================================


@pytest.mark.unit
def test_no_server_scope_backend_url_default_in_conf_templates():
    """A server-scope `set $backend_url "";` re-runs inside the /validate
    auth_request subrequest (subrequests share the parent's variable array and
    re-run the server rewrite phase), blanking the per-server upstream the
    /mcp-proxy/ location set. That makes /validate forward an empty
    X-Resolved-Upstream, skip minting the internal token, and 401 every MCP
    call with "Missing internal proxy token". $backend_url must be set ONLY in
    the generated /mcp-proxy/ location blocks -- never at server scope.
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    conf_files = [
        repo_root / "docker" / "nginx_rev_proxy_http_and_https.conf",
        repo_root / "docker" / "nginx_rev_proxy_http_only.conf",
    ]
    # Match an actual directive (line starting with optional whitespace then
    # `set $backend_url`), NOT the explanatory NOTE comments (which start with #).
    directive = re.compile(r"^\s*set\s+\$backend_url\b", re.MULTILINE)
    for conf in conf_files:
        text = conf.read_text()
        offenders = [
            line
            for line in text.splitlines()
            if directive.match(line) and not line.lstrip().startswith("#")
        ]
        assert not offenders, (
            f"{conf.name} contains a `set $backend_url` directive outside the "
            f"generated /mcp-proxy/ blocks: {offenders}. This re-runs in the "
            f"/validate subrequest and breaks internal-token minting. See the "
            f"NOTE comment in the server block."
        )


def test_backend_url_declared_at_http_scope():
    """$backend_url is referenced in the shared `location = /validate` block but only
    `set` inside the generated /mcp-proxy/ blocks. When no such block renders (no healthy
    MCP server / registry-only mode), an undeclared variable makes nginx fail to START
    with `[emerg] unknown "backend_url" variable`. It must be DECLARED at http scope via
    a `map ... { default ""; }` (clobber-safe, unlike a server-scope `set`)."""
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    conf_files = [
        repo_root / "docker" / "nginx_rev_proxy_http_and_https.conf",
        repo_root / "docker" / "nginx_rev_proxy_http_only.conf",
    ]
    backend_map = re.compile(r"^\s*map\s+\$\S+\s+\$backend_url\s*\{", re.MULTILINE)
    for conf in conf_files:
        text = conf.read_text()
        assert backend_map.search(text), (
            f'{conf.name} is missing the http-scope `map ... $backend_url {{ default ""; }}` '
            f"declaration. Without it nginx fails to start when no /mcp-proxy/ block renders."
        )


def test_registry_api_auth_declared_at_http_scope():
    """$registry_api_auth has the same requirement as $backend_url: it is referenced in
    the shared /validate block but only `set` inside the /api/ location blocks, so it must
    be declared at http scope via a `map ... { default ""; }` or nginx fails to start on
    requests that don't render an /api/ block."""
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    conf_files = [
        repo_root / "docker" / "nginx_rev_proxy_http_and_https.conf",
        repo_root / "docker" / "nginx_rev_proxy_http_only.conf",
    ]
    api_auth_map = re.compile(r"^\s*map\s+\$\S+\s+\$registry_api_auth\s*\{", re.MULTILINE)
    for conf in conf_files:
        text = conf.read_text()
        assert api_auth_map.search(text), (
            f'{conf.name} is missing the http-scope `map ... $registry_api_auth {{ default ""; }}` '
            f"declaration. Without it nginx fails to start on non-API requests."
        )


def test_no_server_scope_registry_api_auth_default_in_conf_templates():
    """Same footgun as $backend_url: a server-scope `set $registry_api_auth "";`
    would re-run inside the /validate auth_request subrequest and blank the marker
    the /api/ location set, so /validate would never mint the registry-UI token.
    The marker must be set ONLY inside the protected /api/ location blocks.
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    conf_files = [
        repo_root / "docker" / "nginx_rev_proxy_http_and_https.conf",
        repo_root / "docker" / "nginx_rev_proxy_http_only.conf",
    ]
    directive = re.compile(r'^\s*set\s+\$registry_api_auth\s+""', re.MULTILINE)
    for conf in conf_files:
        text = conf.read_text()
        offenders = [
            line
            for line in text.splitlines()
            if directive.match(line) and not line.lstrip().startswith("#")
        ]
        assert not offenders, (
            f'{conf.name} contains a `set $registry_api_auth ""` default outside '
            f"the protected /api/ blocks: {offenders}. This re-runs in the /validate "
            f"subrequest and breaks registry-UI token minting."
        )


def test_registry_api_locations_forward_internal_token():
    """Every protected registry-API location in the static confs sets the marker
    and forwards the registry-UI token; the shared /validate block forwards the
    marker. Guards Phase 4 against a partial wiring that would 401 /api/ calls.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    conf_files = [
        repo_root / "docker" / "nginx_rev_proxy_http_and_https.conf",
        repo_root / "docker" / "nginx_rev_proxy_http_only.conf",
    ]
    for conf in conf_files:
        text = conf.read_text()
        # The count of protected registry-API locations equals the count of
        # `auth_request /validate;` directives (only registry-API locations use it).
        n_protected = text.count("auth_request /validate;")
        assert n_protected > 0, f"{conf.name}: expected protected /api/ locations"
        # Marker set + token capture + token forward appear once per protected location.
        assert text.count('set $registry_api_auth "1";') == n_protected, conf.name
        assert (
            text.count(
                "auth_request_set $auth_internal_token_registry "
                "$upstream_http_x_internal_token_registry;"
            )
            == n_protected
        ), conf.name
        assert (
            text.count("proxy_set_header X-Internal-Token-Registry $auth_internal_token_registry;")
            == n_protected
        ), conf.name
        # The shared /validate block forwards the marker (once per server block).
        assert "proxy_set_header X-Registry-Api-Auth $registry_api_auth;" in text, conf.name


def test_generated_protected_api_block_carries_internal_token():
    """The generated /api/ block (protected) carries the marker + token forward;
    the unprotected variant (NGINX_DISABLE_API_AUTH_REQUEST) carries none.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[3] / "registry" / "core" / "nginx_service.py"
    ).read_text()
    # The protected block string must contain all three directives.
    assert 'set $registry_api_auth "1";' in src
    assert (
        "auth_request_set $auth_internal_token_registry "
        "$upstream_http_x_internal_token_registry;" in src
    )
    assert "proxy_set_header X-Internal-Token-Registry $auth_internal_token_registry;" in src


@pytest.mark.unit
class TestResolveMcpProxyReadTimeout:
    """Tests for the nginx MCP proxy_read_timeout derivation helper.

    The nginx read/send timeout for the /mcp-proxy/ location blocks is derived
    from settings.mcp_proxy_timeout (MCP_PROXY_TIMEOUT) plus a fixed buffer, so
    the inner auth-server hop always times out first. Credit: derivation
    approach contributed by @go-faustino (PR #1321).
    """

    def test_default_is_upstream_plus_buffer(self):
        """Default 30s upstream timeout yields 60s (30s + 30s buffer)."""
        from registry.core import nginx_service as ns

        fake_settings = MagicMock(mcp_proxy_timeout=30.0)
        with patch.object(ns, "settings", fake_settings):
            assert ns._resolve_mcp_proxy_read_timeout_seconds() == 60

    def test_raised_upstream_scales(self):
        """A raised upstream timeout scales the nginx read timeout with headroom."""
        from registry.core import nginx_service as ns

        fake_settings = MagicMock(mcp_proxy_timeout=300.0)
        with patch.object(ns, "settings", fake_settings):
            assert ns._resolve_mcp_proxy_read_timeout_seconds() == 330

    def test_fractional_upstream_rounds_up(self):
        """A fractional upstream timeout is rounded up before adding the buffer."""
        from registry.core import nginx_service as ns

        fake_settings = MagicMock(mcp_proxy_timeout=45.5)
        with patch.object(ns, "settings", fake_settings):
            assert ns._resolve_mcp_proxy_read_timeout_seconds() == 76

    def test_invalid_value_falls_back_to_default(self):
        """A non-numeric value falls back to the 30s default (-> 60s)."""
        from registry.core import nginx_service as ns

        fake_settings = MagicMock(mcp_proxy_timeout="not-a-float")
        with patch.object(ns, "settings", fake_settings):
            assert ns._resolve_mcp_proxy_read_timeout_seconds() == 60

    def test_none_value_falls_back_to_default(self):
        """A missing (None) value falls back to the 30s default (-> 60s)."""
        from registry.core import nginx_service as ns

        fake_settings = MagicMock(mcp_proxy_timeout=None)
        with patch.object(ns, "settings", fake_settings):
            assert ns._resolve_mcp_proxy_read_timeout_seconds() == 60


# =============================================================================
# CONFIG-INJECTION DEFENSE: proxy_pass_url is escaped at interpolation
# =============================================================================


@pytest.mark.unit
def test_location_block_escapes_backend_url_metacharacters(nginx_service):
    """A crafted proxy_pass_url cannot break out of the quoted set directive.

    Registration validation already rejects nginx metacharacters, but the
    location-block builder escapes defensively so legacy/persisted values still
    cannot inject nginx directives.
    """
    malicious = 'http://evil.com/";}\nlocation /x { proxy_pass http://attacker;'

    block = nginx_service._create_location_block(
        path="/evil",
        proxy_pass_url=malicious,
        transport_type="streamable-http",
        server_info={"server_name": "evil"},
    )

    # The raw injection payload must not appear verbatim: quotes/backslashes are
    # escaped and newlines collapsed, so the "; }" directive terminator cannot
    # close the `set $backend_url "..."` string context.
    assert 'set $backend_url "http://evil.com/";}' not in block
    # A real newline must not survive inside the generated directive value.
    assert "\nlocation /x { proxy_pass http://attacker;" not in block
    # The escaped form (backslash-quote) is what lands in the config.
    assert '\\"' in block


@pytest.mark.unit
def test_sanitize_for_nginx_set_escapes_quotes_and_newlines():
    """The nginx sanitizer escapes quotes/backslashes and collapses newlines."""
    out = NginxConfigService._sanitize_for_nginx_set('a"b\\c\nd')
    assert '"' not in out.replace('\\"', "")  # only escaped quotes remain
    assert "\n" not in out


# =============================================================================
# Config validation before persistence (cold-start DoS protection)
# =============================================================================
#
# A malformed generated config must never overwrite the live config on disk:
# even though a bad reload is skipped, a broken file on disk takes down the
# NEXT nginx cold start (container restart), downing the whole gateway. These
# tests exercise the real _write_and_validate_config flow (the autouse fixture
# stubs the module attribute, so we call the captured real function directly).


class TestConfigValidationBeforePersist:
    @pytest.fixture(autouse=True)
    def _real_atomic_write(self):
        """Restore the real _atomic_write_text so these tests actually write to
        the tmp_path (the module-level autouse fixture stubs it), and make the
        nginx binary appear present by default so the validation branch runs.
        Tests that model a missing binary re-patch shutil.which themselves."""
        with (
            patch.object(nginx_module, "_atomic_write_text", _REAL_ATOMIC_WRITE),
            patch("shutil.which", return_value="/usr/sbin/nginx"),
        ):
            yield

    def test_valid_config_is_promoted(self, tmp_path):
        """A candidate that passes nginx -t replaces the live config."""
        live = tmp_path / "nginx_rev_proxy.conf"
        live.write_text("# last good\n")

        with patch.object(nginx_module, "_run_nginx_config_test", return_value=(True, "")):
            _REAL_WRITE_AND_VALIDATE(live, "# new valid config\n")

        assert live.read_text() == "# new valid config\n"
        # No backup left behind on success.
        assert not (tmp_path / (live.name + nginx_module._LAST_GOOD_SUFFIX)).exists()

    def test_invalid_config_does_not_overwrite_live_config(self, tmp_path):
        """A candidate rejected by nginx -t must not persist; the last-good
        config is restored so a subsequent cold start still boots."""
        live = tmp_path / "nginx_rev_proxy.conf"
        live.write_text("# last good\n")

        with patch.object(
            nginx_module,
            "_run_nginx_config_test",
            return_value=(False, "nginx: [emerg] unexpected }"),
        ):
            with pytest.raises(RuntimeError, match="nginx config rejected"):
                _REAL_WRITE_AND_VALIDATE(live, "# broken }\ninjected junk\n")

        # The broken candidate never survives on disk: last-good is intact.
        assert live.read_text() == "# last good\n"
        # No stray backup file remains.
        assert not (tmp_path / (live.name + nginx_module._LAST_GOOD_SUFFIX)).exists()

    def test_invalid_config_with_no_prior_config_leaves_nothing_on_disk(self, tmp_path):
        """When there is no prior live config, a rejected candidate is removed
        entirely so a cold start does not load a broken file."""
        live = tmp_path / "nginx_rev_proxy.conf"
        assert not live.exists()

        with patch.object(nginx_module, "_run_nginx_config_test", return_value=(False, "bad")):
            with pytest.raises(RuntimeError, match="nginx config rejected"):
                _REAL_WRITE_AND_VALIDATE(live, "# broken\n")

        assert not live.exists()

    def test_missing_nginx_binary_promotes_without_validation(self, tmp_path):
        """On a single-container host with no nginx (default), the candidate is
        promoted without a test — there is no nginx cold start to protect."""
        live = tmp_path / "nginx_rev_proxy.conf"
        live.write_text("# last good\n")

        settings_stub = MagicMock()
        settings_stub.nginx_config_validation_required = False
        settings_stub.nginx_updates_enabled = True
        with (
            patch.object(nginx_module, "settings", settings_stub),
            patch("shutil.which", return_value=None),
        ):
            _REAL_WRITE_AND_VALIDATE(live, "# new config, unvalidatable\n")

        assert live.read_text() == "# new config, unvalidatable\n"

    def test_missing_nginx_binary_fails_closed_when_validation_required(self, tmp_path):
        """In a split/sidecar topology (nginx in another container) the operator
        sets nginx_config_validation_required=True. When the registry cannot run
        nginx -t, an unvalidated config must NOT be promoted — and the candidate
        must never even touch the live path (no TOCTOU window for the sidecar)."""
        live = tmp_path / "nginx_rev_proxy.conf"
        live.write_text("# last good\n")

        settings_stub = MagicMock()
        settings_stub.nginx_config_validation_required = True
        with (
            patch.object(nginx_module, "settings", settings_stub),
            patch("shutil.which", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="nginx config rejected"):
                _REAL_WRITE_AND_VALIDATE(live, "# unvalidatable candidate\n")

        # Last-good config is intact; the unvalidated candidate never reached disk.
        assert live.read_text() == "# last good\n"


class TestRunNginxConfigTest:
    def test_returns_no_binary_sentinel_when_nginx_absent(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            passed, message = nginx_module._run_nginx_config_test()
        assert passed is False
        assert message == nginx_module._NGINX_TEST_NO_BINARY

    def test_returns_false_on_nonzero_exit(self):
        result = MagicMock()
        result.returncode = 1
        result.stderr = "nginx: [emerg] boom"
        with patch("subprocess.run", return_value=result):
            passed, message = nginx_module._run_nginx_config_test()
        assert passed is False
        assert "boom" in message

    def test_returns_false_on_timeout(self):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nginx", 5)):
            passed, message = nginx_module._run_nginx_config_test()
        assert passed is False
        assert "timed out" in message

    def test_returns_true_on_success(self):
        result = MagicMock()
        result.returncode = 0
        result.stderr = "syntax is ok"
        with patch("subprocess.run", return_value=result):
            passed, _message = nginx_module._run_nginx_config_test()
        assert passed is True


def _scheduler_settings(live):
    """A settings stub exposing just the attributes the scheduler reload path
    reads, so the read-only nginx_config_path property can be redirected."""
    s = MagicMock()
    s.nginx_config_path = live
    s.nginx_updates_enabled = True
    s.deployment_mode = MagicMock()
    s.deployment_mode.value = "with-gateway"
    return s


@pytest.mark.asyncio
async def test_scheduler_still_reloads_valid_config(monkeypatch, tmp_path):
    """The debounced scheduler applies a changed, valid config and advances its
    last-applied hash (validation is wired in without breaking the reload)."""
    from registry.core.nginx_service import NginxReloadScheduler

    scheduler = NginxReloadScheduler(debounce_seconds=0.0, poll_external=False)

    live = tmp_path / "nginx_rev_proxy.conf"

    async def _fake_fetch():
        return {}

    monkeypatch.setattr(nginx_module, "settings", _scheduler_settings(live))
    monkeypatch.setattr(nginx_module, "_atomic_write_text", _REAL_ATOMIC_WRITE)
    monkeypatch.setattr(nginx_module, "_write_and_validate_config", _REAL_WRITE_AND_VALIDATE)
    monkeypatch.setattr(nginx_module, "_fetch_all_enabled_servers", _fake_fetch)
    monkeypatch.setattr(
        nginx_module.nginx_service, "render_config", AsyncMock(return_value="# valid config\n")
    )
    monkeypatch.setattr(nginx_module.nginx_service, "_commit_virtual_server_mappings", AsyncMock())
    # nginx binary present; nginx -t accepts the candidate; reload succeeds.
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/sbin/nginx")
    monkeypatch.setattr(nginx_module, "_run_nginx_config_test", lambda: (True, ""))
    monkeypatch.setattr(nginx_module.nginx_service, "reload_nginx", MagicMock(return_value=True))

    await scheduler._do_reload_if_changed()

    assert live.read_text() == "# valid config\n"
    assert scheduler._last_config_hash != ""


@pytest.mark.asyncio
async def test_scheduler_rejects_invalid_config_and_keeps_last_good(monkeypatch, tmp_path):
    """When nginx -t rejects the rendered config, the scheduler must NOT
    overwrite the live config, must NOT advance its hash, and must stay dirty
    for a later retry."""
    from registry.core.nginx_service import NginxReloadScheduler

    scheduler = NginxReloadScheduler(debounce_seconds=0.0, poll_external=False)

    live = tmp_path / "nginx_rev_proxy.conf"
    live.write_text("# last good\n")

    async def _fake_fetch():
        return {}

    monkeypatch.setattr(nginx_module, "settings", _scheduler_settings(live))
    monkeypatch.setattr(nginx_module, "_atomic_write_text", _REAL_ATOMIC_WRITE)
    monkeypatch.setattr(nginx_module, "_write_and_validate_config", _REAL_WRITE_AND_VALIDATE)
    monkeypatch.setattr(nginx_module, "_fetch_all_enabled_servers", _fake_fetch)
    monkeypatch.setattr(
        nginx_module.nginx_service, "render_config", AsyncMock(return_value="# broken }\n")
    )
    monkeypatch.setattr(nginx_module.nginx_service, "_commit_virtual_server_mappings", AsyncMock())
    # nginx binary present; nginx -t rejects the candidate.
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/sbin/nginx")
    monkeypatch.setattr(nginx_module, "_run_nginx_config_test", lambda: (False, "emerg"))
    reload_mock = MagicMock(return_value=True)
    monkeypatch.setattr(nginx_module.nginx_service, "reload_nginx", reload_mock)

    await scheduler._do_reload_if_changed()

    # Live config untouched, hash never advanced (stays initial ""), reload
    # never issued, dirty stays set for a later retry.
    assert live.read_text() == "# last good\n"
    assert scheduler._last_config_hash == ""
    reload_mock.assert_not_called()
    assert scheduler._dirty is True
