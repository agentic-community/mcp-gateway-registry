"""
Unit tests for registry/core/nginx_service.py

Tests the NginxConfigService for configuration generation and reload.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import httpx
import pytest

from registry.constants import HealthStatus
from registry.core.nginx_service import NginxConfigService


# =============================================================================
# TEST FIXTURES
# =============================================================================


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
    assert "proxy_pass http://localhost:8000/mcp" in blocks[0]


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
    assert "proxy_pass http://localhost:8000/sse" in blocks[0]


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
    assert "proxy_pass http://localhost:8000/mcp" in block
    assert "proxy_buffering off" in block
    assert "auth_request /validate" in block


@pytest.mark.unit
def test_create_location_block_sse(nginx_service):
    """Test creating location block for SSE."""
    block = nginx_service._create_location_block("/test", "http://localhost:8000/sse", "sse")

    assert "location {{ROOT_PATH}}/test" in block
    assert "proxy_pass http://localhost:8000/sse" in block
    assert "proxy_buffering off" in block
    assert "proxy_set_header Connection $http_connection" in block


@pytest.mark.unit
def test_create_location_block_external_service(nginx_service):
    """Test creating location block for external HTTPS service."""
    block = nginx_service._create_location_block(
        "/test", "https://api.example.com/mcp", "streamable-http"
    )

    assert "location {{ROOT_PATH}}/test" in block
    assert "proxy_pass https://api.example.com/mcp" in block
    # Should use upstream hostname for external services
    assert "proxy_set_header Host api.example.com" in block


@pytest.mark.unit
def test_create_location_block_internal_service(nginx_service):
    """Test creating location block for internal service."""
    block = nginx_service._create_location_block(
        "/test", "http://backend:8000/mcp", "streamable-http"
    )

    assert "location {{ROOT_PATH}}/test" in block
    assert "proxy_pass http://backend:8000/mcp" in block
    # Should preserve original host for internal services
    assert "proxy_set_header Host $host" in block


@pytest.mark.unit
def test_create_location_block_direct_transport(nginx_service):
    """Test creating location block for direct transport."""
    block = nginx_service._create_location_block("/test", "http://localhost:8000", "direct")

    assert "location {{ROOT_PATH}}/test" in block
    assert "proxy_pass http://localhost:8000" in block
    assert "proxy_cache off" in block


# =============================================================================
# KEYCLOAK CONFIGURATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_keycloak_parsing(
    nginx_service, sample_servers, mock_health_service
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
                            write_calls = list(mock_file().write.call_args_list)
                            assert len(write_calls) > 0
                            written_content = write_calls[0][0][0]
                            assert "keycloak.example.com" in written_content
                            assert "8443" in written_content


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
    nginx_service, sample_servers, mock_health_service
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
                            write_calls = mock_file().write.call_args_list
                            assert len(write_calls) > 0
                            written_content = write_calls[0][0][0]
                            assert "/keycloak/" not in written_content
                            assert "/realms/" not in written_content
                            assert "KEYCLOAK_LOCATIONS_START" not in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_keeps_keycloak_locations_for_keycloak(
    nginx_service, sample_servers, mock_health_service
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
                            write_calls = mock_file().write.call_args_list
                            assert len(write_calls) > 0
                            written_content = write_calls[0][0][0]
                            assert "/keycloak/" in written_content
                            assert "/realms/" in written_content
                            assert "keycloak.example.com" in written_content
                            assert "8443" in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_strips_keycloak_locations_for_cognito(
    nginx_service, sample_servers, mock_health_service
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

                            write_calls = mock_file().write.call_args_list
                            assert len(write_calls) > 0
                            written_content = write_calls[0][0][0]
                            assert "/keycloak/" not in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_keycloak_https_default_port(
    nginx_service, sample_servers, mock_health_service
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
                            written_content = mock_file().write.call_args_list[0][0][0]
                            assert "https" in written_content
                            assert "443" in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_keycloak_hostname_fallback(
    nginx_service, sample_servers, mock_health_service
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
                            written_content = mock_file().write.call_args_list[0][0][0]
                            # Should still contain keycloak as the host (netloc fallback)
                            assert "keycloak" in written_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_config_async_keycloak_url_parse_exception(
    nginx_service, sample_servers, mock_health_service
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
                                written_content = mock_file().write.call_args_list[0][0][0]
                                # Should fall back to defaults
                                assert "http" in written_content
                                assert "keycloak" in written_content
                                assert "8080" in written_content
