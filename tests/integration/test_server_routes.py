"""
Integration tests for server routes.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from tests.fixtures.factories import ServerInfoFactory


@pytest.mark.integration
@pytest.mark.servers
class TestServerRoutes:
    """Integration tests for server management routes."""

    def test_dashboard_unauthorized(self, test_client: TestClient):
        """Test dashboard access without authentication (API UI entrypoint)."""
        response = test_client.get("/api/", follow_redirects=False)
        assert response.status_code in [307, 302]

    def test_dashboard_authorized(self, test_client: TestClient, mock_authenticated_user):
        """Test dashboard access with authentication (API UI entrypoint)."""
        from registry.core.config import settings

        user_context = {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "list_service": ["all"],
                "register_service": ["all"],
                "toggle_service": ["all"],
                "health_check_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        with patch("registry.api.server_routes.enhanced_auth", return_value=user_context), patch(
            "registry.api.server_routes.server_service"
        ) as mock_service:
            mock_service.get_all_servers.return_value = {}
            mock_service.is_service_enabled.return_value = False
            
            response = test_client.get(
                "/api/",
                cookies={
                    settings.session_cookie_name: "dummy-session",
                },
            )
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_register_server_success(self, test_client: TestClient, mock_authenticated_user):
        """Test successful server registration."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "register_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        server_data = ServerInfoFactory()
        
        with patch('registry.api.server_routes.server_service') as mock_service, \
             patch('registry.search.service.faiss_service') as mock_faiss, \
             patch('registry.core.nginx_service.nginx_service') as mock_nginx, \
             patch('registry.health.service.health_service') as mock_health:
            
            mock_service.register_server.return_value = True
            mock_faiss.add_or_update_service = AsyncMock()
            mock_nginx.generate_config_async = AsyncMock()
            mock_health.broadcast_health_update = AsyncMock()
            mock_service.get_enabled_services.return_value = []
            mock_service.get_server_info.return_value = None
            
            response = test_client.post("/api/register", data={
                "name": server_data["server_name"],
                "description": server_data["description"],
                "path": server_data["path"],
                "proxy_pass_url": server_data["proxy_pass_url"],
                "tags": ",".join(server_data["tags"]),
                "num_tools": server_data["num_tools"],
                "num_stars": server_data["num_stars"],
                "is_python": server_data["is_python"],
                "license": server_data["license"],
            })
            
            assert response.status_code == 201
            data = response.json()
            assert data["message"] == "Service registered successfully"
            assert data["service"]["server_name"] == server_data["server_name"]

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_register_server_duplicate_path(self, test_client: TestClient, mock_authenticated_user):
        """Test registering server with duplicate path."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "register_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        server_data = ServerInfoFactory()
        
        with patch('registry.api.server_routes.server_service') as mock_service:
            mock_service.register_server.return_value = False
            
            response = test_client.post("/api/register", data={
                "name": server_data["server_name"],
                "description": server_data["description"],
                "path": server_data["path"],
                "proxy_pass_url": server_data["proxy_pass_url"],
            })
            
            assert response.status_code == 400
            data = response.json()
            assert "already exists" in data["error"]

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_toggle_service_success(self, test_client: TestClient, mock_authenticated_user):
        """Test successful service toggle."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "toggle_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        server_data = ServerInfoFactory()
        
        with patch('registry.api.server_routes.server_service') as mock_service, \
             patch('registry.search.service.faiss_service') as mock_faiss, \
             patch('registry.core.nginx_service.nginx_service') as mock_nginx, \
             patch('registry.health.service.health_service') as mock_health:
            
            mock_service.get_server_info.return_value = server_data
            mock_service.toggle_service.return_value = True
            mock_faiss.add_or_update_service = AsyncMock()
            mock_nginx.generate_config_async = AsyncMock()
            mock_health.broadcast_health_update = AsyncMock()
            mock_service.get_enabled_services.return_value = []
            
            response = test_client.post(f"/api/toggle{server_data['path']}", data={
                "enabled": "on"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["service_path"] == server_data["path"]
            assert data["new_enabled_state"] is True

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_toggle_service_not_found(self, test_client: TestClient, mock_authenticated_user):
        """Test toggling non-existent service."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "toggle_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        with patch('registry.api.server_routes.server_service') as mock_service:
            mock_service.get_server_info.return_value = None
            
            response = test_client.post("/api/toggle/nonexistent", data={
                "enabled": "on"
            })
            
            assert response.status_code == 404

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_get_server_details_success(self, test_client: TestClient, mock_authenticated_user):
        """Test getting server details."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "list_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        server_data = ServerInfoFactory()
        
        with patch('registry.api.server_routes.server_service') as mock_service:
            mock_service.get_server_info.return_value = server_data
            
            response = test_client.get(f"/api/server_details{server_data['path']}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["server_name"] == server_data["server_name"]

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_get_server_details_not_found(self, test_client: TestClient, mock_authenticated_user):
        """Test getting details for non-existent server."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "list_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        with patch('registry.api.server_routes.server_service') as mock_service:
            mock_service.get_server_info.return_value = None
            
            response = test_client.get("/api/server_details/nonexistent")
            
            assert response.status_code == 404

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_get_all_server_details(self, test_client: TestClient, mock_authenticated_user):
        """Test getting all server details."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "list_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        servers = {
            "/test1": ServerInfoFactory(),
            "/test2": ServerInfoFactory()
        }
        
        with patch('registry.api.server_routes.server_service') as mock_service:
            mock_service.get_all_servers.return_value = servers
            
            response = test_client.get("/api/server_details/all")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert "/test1" in data
            assert "/test2" in data

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_refresh_service_success(self, test_client: TestClient, mock_authenticated_user):
        """Test refreshing service."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "health_check_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        server_data = ServerInfoFactory()
        
        with patch('registry.api.server_routes.server_service') as mock_service, \
             patch('registry.search.service.faiss_service') as mock_faiss, \
             patch('registry.core.nginx_service.nginx_service') as mock_nginx, \
             patch('registry.health.service.health_service') as mock_health:
            
            mock_service.get_server_info.return_value = server_data
            mock_service.is_service_enabled.return_value = True
            mock_service.get_enabled_services.return_value = []
            mock_faiss.add_or_update_service = AsyncMock()
            mock_nginx.generate_config_async = AsyncMock()
            mock_health.perform_immediate_health_check = AsyncMock(return_value=("healthy", None))
            mock_health.broadcast_health_update = AsyncMock()
            
            response = test_client.post(f"/api/refresh{server_data['path']}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["service_path"] == server_data["path"]
            assert data["status"] == "healthy"

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_refresh_service_not_found(self, test_client: TestClient, mock_authenticated_user):
        """Test refreshing non-existent service."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "health_check_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        with patch('registry.api.server_routes.server_service') as mock_service:
            mock_service.get_server_info.return_value = None
            
            response = test_client.post("/api/refresh/nonexistent")
            
            assert response.status_code == 404

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_edit_server_form_success(self, test_client: TestClient, mock_authenticated_user):
        """Test getting edit server form."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "modify_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        server_data = ServerInfoFactory()
        
        with patch('registry.api.server_routes.server_service') as mock_service:
            mock_service.get_server_info.return_value = server_data
            
            response = test_client.get(f"/api/edit{server_data['path']}")
            
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_edit_server_form_not_found(self, test_client: TestClient, mock_authenticated_user):
        """Test getting edit form for non-existent server."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "modify_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        with patch('registry.api.server_routes.server_service') as mock_service:
            mock_service.get_server_info.return_value = None
            
            response = test_client.get("/api/edit/nonexistent")
            
            assert response.status_code == 404

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_edit_server_submit_success(self, test_client: TestClient, mock_authenticated_user):
        """Test successful server edit submission."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "modify_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        server_data = ServerInfoFactory()
        
        with patch('registry.api.server_routes.server_service') as mock_service, \
             patch('registry.search.service.faiss_service') as mock_faiss, \
             patch('registry.core.nginx_service.nginx_service') as mock_nginx:
            
            mock_service.get_server_info.return_value = server_data
            mock_service.update_server.return_value = True
            mock_service.is_service_enabled.return_value = False
            mock_service.get_enabled_services.return_value = []
            mock_faiss.add_or_update_service = AsyncMock()
            mock_nginx.generate_config_async = AsyncMock()
            
            response = test_client.post(f"/api/edit{server_data['path']}", data={
                "name": "Updated Name",
                "description": server_data["description"],
                "proxy_pass_url": server_data["proxy_pass_url"],
            }, follow_redirects=False)
            
            # Should redirect to main page
            assert response.status_code == 303
            assert response.headers["location"] == "/"

        app.dependency_overrides.pop(enhanced_auth, None)

    def test_edit_server_submit_not_found(self, test_client: TestClient, mock_authenticated_user):
        """Test editing non-existent server."""
        from registry.auth.dependencies import enhanced_auth

        app = test_client.app
        app.dependency_overrides[enhanced_auth] = lambda: {
            "username": "testuser",
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "auth_method": "traditional",
            "provider": "local",
            "accessible_servers": ["all"],
            "accessible_services": ["all"],
            "accessible_agents": ["all"],
            "ui_permissions": {
                "modify_service": ["all"],
            },
            "can_modify_servers": True,
            "is_admin": True,
        }

        with patch('registry.api.server_routes.server_service') as mock_service:
            mock_service.get_server_info.return_value = None
            
            response = test_client.post("/api/edit/nonexistent", data={
                "name": "Test",
                "proxy_pass_url": "http://localhost:8000",
            })
            
            assert response.status_code == 404

        app.dependency_overrides.pop(enhanced_auth, None)
