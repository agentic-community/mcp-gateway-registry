"""
Unit tests for the Nginx configuration service.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, call, AsyncMock
import tempfile
import shutil

from registry.constants import HealthStatus
from registry.core.nginx_service import NginxConfigService
from registry.health.service import health_service


@pytest.mark.unit
@pytest.mark.core
class TestNginxConfigService:
    """Test class for nginx config service."""
    
    @pytest.fixture(autouse=True)
    def setup_patches(self, temp_dir):
        """Set up patches for all tests in this class."""
        # Create mock settings
        mock_settings = Mock()
        mock_settings.nginx_config_path = temp_dir / "nginx.conf"

        template_path = temp_dir / "nginx_template.conf"
        mock_constants = Mock()
        mock_constants.SSL_CERT_PATH = str(temp_dir / "ssl_cert.pem")  # does not exist
        mock_constants.SSL_KEY_PATH = str(temp_dir / "ssl_key.pem")  # does not exist
        mock_constants.NGINX_TEMPLATE_HTTP_ONLY = str(template_path)
        mock_constants.NGINX_TEMPLATE_HTTP_ONLY_LOCAL = str(template_path)
        mock_constants.NGINX_TEMPLATE_HTTP_AND_HTTPS = str(temp_dir / "nginx_https.conf")
        mock_constants.NGINX_TEMPLATE_HTTP_AND_HTTPS_LOCAL = str(temp_dir / "nginx_https.conf")
        mock_constants.ANTHROPIC_API_VERSION = "v0.1"
        
        # Patch settings for the duration of each test
        with patch('registry.core.nginx_service.settings', mock_settings), \
             patch('registry.core.nginx_service.REGISTRY_CONSTANTS', mock_constants), \
            patch.object(NginxConfigService, "get_additional_server_names", new=AsyncMock(return_value="")):
            self.mock_settings = mock_settings
            self.template_path = template_path
            original_health_status = health_service.server_health_status
            healthy_status = Mock()
            healthy_status.get = Mock(return_value=HealthStatus.HEALTHY)
            health_service.server_health_status = healthy_status
            yield
            health_service.server_health_status = original_health_status

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def nginx_service(self):
        """Create a Nginx service instance."""
        template_content = """
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        {{LOCATION_BLOCKS}}
    }
}
"""
        self.template_path.write_text(template_content)
        return NginxConfigService()

    def test_init(self, nginx_service):
        """Test Nginx service initialization."""
        assert nginx_service.nginx_template_path == self.template_path

    def test_generate_config_success(self, nginx_service):
        """Test successful config generation."""
        servers = {
            "/api/server1": {
                "proxy_pass_url": "http://localhost:8001"
            },
            "/api/server2": {
                "proxy_pass_url": "http://localhost:8002"
            }
        }
        
        result = nginx_service.generate_config(servers)
        
        assert result is True
        assert self.mock_settings.nginx_config_path.exists()
        
        # Check generated content
        config_content = self.mock_settings.nginx_config_path.read_text()
        assert "location /api/server1/mcp {" in config_content
        assert "location /api/server2/mcp {" in config_content
        assert "proxy_pass http://localhost:8001/mcp;" in config_content
        assert "proxy_pass http://localhost:8002/mcp;" in config_content

    def test_generate_config_no_template(self, nginx_service):
        """Test config generation when template doesn't exist."""
        nginx_service.nginx_template_path = self.template_path.parent / "missing_template.conf"
        servers = {"/api/test": {"proxy_pass_url": "http://localhost:8001"}}
        
        result = nginx_service.generate_config(servers)
        
        assert result is False

    def test_generate_config_empty_servers(self, nginx_service):
        """Test config generation with empty servers."""
        servers = {}
        
        result = nginx_service.generate_config(servers)
        
        assert result is True
        config_content = self.mock_settings.nginx_config_path.read_text()
        # Should have base structure without location blocks
        assert "events {" in config_content
        assert "http {" in config_content

    def test_generate_config_servers_without_proxy_pass(self, nginx_service):
        """Test config generation with servers missing proxy_pass_url."""
        servers = {
            "/api/server1": {
                "proxy_pass_url": "http://localhost:8001"
            },
            "/api/server2": {
                "name": "Server 2"  # Missing proxy_pass_url
            }
        }
        
        result = nginx_service.generate_config(servers)
        
        assert result is True
        config_content = self.mock_settings.nginx_config_path.read_text()
        assert "location /api/server1/mcp {" in config_content
        assert "location /api/server2 {" not in config_content

    def test_generate_config_template_read_error(self, nginx_service):
        """Test config generation with template read error."""
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            servers = {"/api/test": {"proxy_pass_url": "http://localhost:8001"}}
            
            result = nginx_service.generate_config(servers)
            
            assert result is False

    def test_generate_config_write_error(self, nginx_service):
        """Test config generation with config file write error."""
        servers = {"/api/test": {"proxy_pass_url": "http://localhost:8001"}}
        
        with patch('builtins.open', side_effect=[
            mock_open(read_data="template content {{LOCATION_BLOCKS}}").return_value,
            IOError("Write permission denied")
        ]):
            result = nginx_service.generate_config(servers)
            
            assert result is False

    def test_generate_config_location_block_formatting(self, nginx_service):
        """Test that location blocks are properly formatted."""
        servers = {
            "/api/test": {
                "proxy_pass_url": "http://localhost:8080"
            }
        }
        
        result = nginx_service.generate_config(servers)
        
        assert result is True
        config_content = self.mock_settings.nginx_config_path.read_text()
        
        # Check that all required proxy headers are included
        assert "proxy_set_header Host $host;" in config_content
        assert "proxy_set_header X-Real-IP $remote_addr;" in config_content
        assert "proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;" in config_content
        assert "proxy_set_header X-Forwarded-Proto $scheme;" in config_content
        assert "header_filter_by_lua_file /etc/nginx/lua/filter_tools_list_headers.lua;" in config_content
        assert "body_filter_by_lua_file /etc/nginx/lua/filter_tools_list.lua;" in config_content
        assert "auth_request_set $auth_allowed_tools $upstream_http_x_allowed_tools;" in config_content

    def test_generate_config_includes_upstream_injection_directives(self, nginx_service):
        """Test that upstream injection headers are wired in generated config."""
        servers = {
            "/fininfo": {
                "proxy_pass_url": "http://localhost:8002",
                "upstream_auth": {
                    "type": "api-key",
                    "provider": None,
                    "credential_binding": "user",
                    "injection": {
                        "kind": "header",
                        "header_name": "X-API-Key",
                        "scheme": None,
                    },
                },
            }
        }

        result = nginx_service.generate_config(servers)

        assert result is True
        config_content = self.mock_settings.nginx_config_path.read_text()

        assert "auth_request /__enforceai_validate_" in config_content
        assert "location = /__enforceai_validate_" in config_content
        assert 'proxy_set_header X-EnforceAI-Server-Path "/fininfo";' in config_content
        assert 'proxy_set_header X-EnforceAI-Upstream-Auth-Type "api-key";' in config_content
        assert (
            'proxy_set_header X-EnforceAI-Upstream-Credential-Binding "user";' in config_content
        )
        assert 'set $enforceai_server_path "/fininfo";' in config_content
        assert 'set $enforceai_upstream_auth_type "api-key";' in config_content
        assert 'set $enforceai_upstream_credential_binding "user";' in config_content
        assert "auth_request_set $mcp_principal $upstream_http_x_mcp_principal;" in config_content
        assert (
            "auth_request_set $enforceai_upstream_api_key $upstream_http_x_enforceai_upstream_api_key;"
            in config_content
        )
        assert "proxy_set_header X-MCP-Principal $mcp_principal;" in config_content
        assert "proxy_set_header X-API-Key $enforceai_upstream_api_key;" in config_content
        assert "error_page 424 = @upstream_credentials_required;" in config_content

    def test_generate_config_multiple_servers(self, nginx_service):
        """Test config generation with multiple servers."""
        servers = {
            "/api/auth": {"proxy_pass_url": "http://localhost:3001"},
            "/api/users": {"proxy_pass_url": "http://localhost:3002"},
            "/api/data": {"proxy_pass_url": "http://localhost:3003"}
        }
        
        result = nginx_service.generate_config(servers)
        
        assert result is True
        config_content = self.mock_settings.nginx_config_path.read_text()
        
        # All location blocks should be present
        assert config_content.count("location /api/") == 3
        assert config_content.count("proxy_pass http://localhost:") == 3

    def test_reload_nginx_success(self, nginx_service):
        """Test successful nginx reload."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = [
                Mock(returncode=0),
                Mock(returncode=0),
            ]
            
            result = nginx_service.reload_nginx()
            
            assert result is True
            assert mock_run.call_args_list == [
                call(["nginx", "-t"], capture_output=True, text=True),
                call(["nginx", "-s", "reload"], capture_output=True, text=True),
            ]

    def test_reload_nginx_failure(self, nginx_service):
        """Test nginx reload failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = [
                Mock(returncode=0),
                Mock(returncode=1, stderr="nginx: [error] invalid configuration"),
            ]
            
            result = nginx_service.reload_nginx()
            
            assert result is False

    def test_reload_nginx_not_found(self, nginx_service):
        """Test nginx reload when nginx binary not found."""
        with patch('subprocess.run', side_effect=FileNotFoundError("nginx not found")):
            result = nginx_service.reload_nginx()
            
            assert result is False

    def test_reload_nginx_exception(self, nginx_service):
        """Test nginx reload with unexpected exception."""
        with patch('subprocess.run', side_effect=Exception("Unexpected error")):
            result = nginx_service.reload_nginx()
            
            assert result is False

    def test_logging_behavior(self, nginx_service):
        """Test that appropriate logging occurs."""
        with patch('registry.core.nginx_service.logger') as mock_logger:
            servers = {"/api/test": {"proxy_pass_url": "http://localhost:8001"}}
            
            nginx_service.generate_config(servers)
            
            # Should log successful generation
            mock_logger.info.assert_called()
            assert any("Generated Nginx configuration" in str(call) 
                      for call in mock_logger.info.call_args_list)

    def test_logging_template_not_found(self, nginx_service):
        """Test logging when template is not found."""
        nginx_service.nginx_template_path = self.template_path.parent / "missing_template.conf"
        with patch('registry.core.nginx_service.logger') as mock_logger:
            servers = {"/api/test": {"proxy_pass_url": "http://localhost:8001"}}
            
            nginx_service.generate_config(servers)
            
            # Should log warning about missing template
            mock_logger.warning.assert_called()
            assert any("Nginx template not found" in str(call) 
                      for call in mock_logger.warning.call_args_list)

    def test_logging_generation_error(self, nginx_service):
        """Test logging when config generation fails."""
        with patch('registry.core.nginx_service.logger') as mock_logger, \
             patch('builtins.open', side_effect=Exception("Test error")):
            
            servers = {"/api/test": {"proxy_pass_url": "http://localhost:8001"}}
            
            nginx_service.generate_config(servers)
            
            # Should log error
            mock_logger.error.assert_called()

    def test_logging_reload_success(self, nginx_service):
        """Test logging for successful nginx reload."""
        with patch('registry.core.nginx_service.logger') as mock_logger, \
             patch('subprocess.run') as mock_run:
            
            mock_run.side_effect = [
                Mock(returncode=0),
                Mock(returncode=0),
            ]
            
            nginx_service.reload_nginx()
            
            # Should log successful reload
            mock_logger.info.assert_called_with("Nginx configuration reloaded successfully")

    def test_logging_reload_failure(self, nginx_service):
        """Test logging for nginx reload failure."""
        with patch('registry.core.nginx_service.logger') as mock_logger, \
             patch('subprocess.run') as mock_run:
            
            mock_run.side_effect = [
                Mock(returncode=0),
                Mock(returncode=1, stderr="Test error"),
            ]
            
            nginx_service.reload_nginx()
            
            # Should log error
            mock_logger.error.assert_called()

    def test_logging_nginx_not_found(self, nginx_service):
        """Test logging when nginx binary is not found."""
        with patch('registry.core.nginx_service.logger') as mock_logger, \
             patch('subprocess.run', side_effect=FileNotFoundError()):
            
            nginx_service.reload_nginx()
            
            # Should log warning
            mock_logger.warning.assert_called_with("Nginx not found - skipping reload")

    def test_template_placeholder_replacement(self, nginx_service, temp_dir):
        """Test that template placeholder is correctly replaced."""
        # Create template with placeholder
        template_content = "upstream servers {\n{{LOCATION_BLOCKS}}\n}"
        template_path = temp_dir / "nginx_template.conf"
        template_path.write_text(template_content)
        
        servers = {
            "/api/test": {"proxy_pass_url": "http://localhost:8001"}
        }
        
        result = nginx_service.generate_config(servers)
        
        assert result is True
        config_content = self.mock_settings.nginx_config_path.read_text()
        
        # Placeholder should be replaced with location block
        assert "{{LOCATION_BLOCKS}}" not in config_content
        assert "location /api/test/mcp {" in config_content

    def test_path_normalization(self, nginx_service):
        """Test that server paths are handled correctly."""
        servers = {
            "/api/test/": {"proxy_pass_url": "http://localhost:8001"},  # trailing slash
            "api/test2": {"proxy_pass_url": "http://localhost:8002"},   # no leading slash
            "/api/test3//": {"proxy_pass_url": "http://localhost:8003"} # double slash
        }
        
        result = nginx_service.generate_config(servers)
        
        assert result is True
        config_content = self.mock_settings.nginx_config_path.read_text()
        
        # Gateway exposes normalized /mcp endpoints for streamable-http.
        assert "location /api/test/mcp {" in config_content
        assert "location /api/test2/mcp {" in config_content
        assert "location /api/test3/mcp {" in config_content

    def test_oauth2_upstream_auth_injects_authorization_header(self, nginx_service):
        servers = {
            "/fininfo/": {
                "proxy_pass_url": "http://localhost:8001",
                "upstream_auth": {
                    "mode": "gateway-managed",
                    "type": "oauth2",
                    "provider": "github",
                    "credential_binding": "user",
                    "injection": {
                        "kind": "header",
                        "header_name": "Authorization",
                        "scheme": "Bearer",
                    },
                },
            },
        }

        result = nginx_service.generate_config(servers)

        assert result is True
        config_content = self.mock_settings.nginx_config_path.read_text()
        assert 'set $enforceai_upstream_auth_type "oauth2";' in config_content
        assert 'set $enforceai_upstream_scheme "Bearer";' in config_content
        assert "proxy_set_header Authorization $enforceai_upstream_authorization;" in config_content

    def test_oauth2_upstream_auth_defaults_header_and_scheme(self, nginx_service):
        servers = {
            "/fininfo/": {
                "proxy_pass_url": "http://localhost:8001",
                "upstream_auth": {
                    "mode": "gateway-managed",
                    "type": "oauth2",
                    "provider": "github",
                    "credential_binding": "user",
                },
            },
        }

        result = nginx_service.generate_config(servers)

        assert result is True
        config_content = self.mock_settings.nginx_config_path.read_text()
        assert 'set $enforceai_upstream_header_name "Authorization";' in config_content
        assert 'set $enforceai_upstream_scheme "Bearer";' in config_content
        assert "proxy_set_header Authorization $enforceai_upstream_authorization;" in config_content

    def test_generate_config_sse_transport_exposes_sse_endpoint(self, nginx_service):
        servers = {
            "/fininfo": {
                "proxy_pass_url": "http://localhost:8009",
                "supported_transports": ["sse"],
            },
        }

        result = nginx_service.generate_config(servers)

        assert result is True
        config_content = self.mock_settings.nginx_config_path.read_text()
        assert "location /fininfo/sse {" in config_content
        assert "proxy_pass http://localhost:8009/sse;" in config_content

    def test_generate_config_multiple_transports_exposes_both_endpoints(self, nginx_service):
        servers = {
            "/both": {
                "proxy_pass_url": "http://localhost:8010",
                "supported_transports": ["streamable-http", "sse"],
            },
        }

        result = nginx_service.generate_config(servers)

        assert result is True
        config_content = self.mock_settings.nginx_config_path.read_text()
        assert "location /both/mcp {" in config_content
        assert "proxy_pass http://localhost:8010/mcp;" in config_content
        assert "location /both/sse {" in config_content
        assert "proxy_pass http://localhost:8010/sse;" in config_content
