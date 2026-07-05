"""
Unit tests for registry/api/peer_management_routes.py

Tests the peer management endpoints including:
- PATCH /api/peers/{peer_id}/token - Update federation token
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from registry.schemas.peer_federation_schema import PeerRegistryConfig

logger = logging.getLogger(__name__)


# =============================================================================
# AUTH MOCK FIXTURES
# =============================================================================


@pytest.fixture
def admin_user_context() -> dict[str, Any]:
    """Create admin user context with peer management permissions."""
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-servers-unrestricted/read", "mcp-servers-unrestricted/execute"],
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "accessible_agents": ["all"],
        "ui_permissions": {
            "list_service": ["all"],
            "toggle_service": ["all"],
            "register_service": ["all"],
            "view_tools": ["all"],
            "refresh_service": ["all"],
            "modify_service": ["all"],
        },
        "auth_method": "session",
    }


@pytest.fixture
def non_admin_user_context() -> dict[str, Any]:
    """Create non-admin user context without peer management permissions."""
    return {
        "username": "regular_user",
        "is_admin": False,
        "groups": ["mcp-users"],
        "scopes": [],
        "accessible_servers": [],
        "accessible_services": [],
        "accessible_agents": [],
        "ui_permissions": {},
        "auth_method": "session",
    }


@pytest.fixture
def mock_auth_admin(admin_user_context):
    """Mock authentication dependencies with admin user."""
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    def mock_nginx_proxied_auth_override():
        return admin_user_context

    app.dependency_overrides[nginx_proxied_auth] = mock_nginx_proxied_auth_override

    yield admin_user_context

    app.dependency_overrides.clear()


@pytest.fixture
def mock_auth_regular(non_admin_user_context):
    """Mock authentication dependencies with regular user."""
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    def mock_nginx_proxied_auth_override():
        return non_admin_user_context

    app.dependency_overrides[nginx_proxied_auth] = mock_nginx_proxied_auth_override

    yield non_admin_user_context

    app.dependency_overrides.clear()


# =============================================================================
# MOCK SERVICE FIXTURES
# =============================================================================


@pytest.fixture
def mock_peer_federation_service():
    """Create mock peer federation service."""
    mock = AsyncMock()
    mock.get_peer_by_id = AsyncMock()
    mock.update_peer = AsyncMock()
    return mock


@pytest.fixture
def sample_peer_config():
    """Sample peer config for testing."""
    return PeerRegistryConfig(
        peer_id="test-peer",
        name="Test Peer Registry",
        endpoint="https://peer.example.com",
        enabled=True,
        sync_mode="all",
        sync_interval_minutes=60,
        federation_token="original-token-abc123",
    )


# =============================================================================
# PATCH /api/peers/{peer_id}/token Tests
# =============================================================================


@pytest.mark.unit
class TestUpdatePeerToken:
    """Tests for PATCH /api/peers/{peer_id}/token endpoint."""

    @pytest.mark.asyncio
    async def test_update_peer_token_success(
        self,
        mock_auth_admin,
        mock_peer_federation_service,
        sample_peer_config,
    ):
        """Test successfully updating peer federation token."""
        # Arrange
        from registry.main import app

        client = TestClient(app)

        # Mock service to return updated peer
        updated_peer = sample_peer_config.model_copy()
        updated_peer.federation_token = "new-token-xyz789"
        mock_peer_federation_service.get_peer_by_id.return_value = sample_peer_config
        mock_peer_federation_service.update_peer.return_value = updated_peer

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=mock_peer_federation_service,
        ):
            # Act
            response = client.patch(
                f"/api/peers/{sample_peer_config.peer_id}/token",
                json={"federation_token": "new-token-xyz789"},
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["message"] == "Federation token updated successfully"
            assert data["peer_id"] == sample_peer_config.peer_id

            # Verify service was called correctly
            mock_peer_federation_service.update_peer.assert_called_once_with(
                sample_peer_config.peer_id,
                {"federation_token": "new-token-xyz789"},
            )

    @pytest.mark.asyncio
    async def test_update_peer_token_not_found(
        self,
        mock_auth_admin,
        mock_peer_federation_service,
    ):
        """Test updating token for non-existent peer returns 404."""
        # Arrange
        from registry.main import app

        client = TestClient(app)

        # Mock service to raise ValueError for non-existent peer
        mock_peer_federation_service.get_peer_by_id.return_value = None
        mock_peer_federation_service.update_peer.side_effect = ValueError(
            "Peer not found: nonexistent-peer"
        )

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=mock_peer_federation_service,
        ):
            # Act
            response = client.patch(
                "/api/peers/nonexistent-peer/token",
                json={"federation_token": "new-token"},
            )

            # Assert
            assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_update_peer_token_requires_admin(
        self,
        mock_auth_regular,
        mock_peer_federation_service,
    ):
        """Test that updating peer token requires admin permissions."""
        # Arrange
        from registry.main import app

        client = TestClient(app)

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=mock_peer_federation_service,
        ):
            # Act
            response = client.patch(
                "/api/peers/test-peer/token",
                json={"federation_token": "new-token"},
            )

            # Assert
            assert response.status_code == status.HTTP_403_FORBIDDEN

            # Verify service was not called
            mock_peer_federation_service.update_peer.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_peer_token_missing_token_field(
        self,
        mock_auth_admin,
        mock_peer_federation_service,
    ):
        """Test that request without federation_token field returns 422."""
        # Arrange
        from registry.main import app

        client = TestClient(app)

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=mock_peer_federation_service,
        ):
            # Act - send empty body
            response = client.patch(
                "/api/peers/test-peer/token",
                json={},
            )

            # Assert
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

            # Verify service was not called
            mock_peer_federation_service.update_peer.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_peer_token_empty_token_value(
        self,
        mock_auth_admin,
        mock_peer_federation_service,
        sample_peer_config,
    ):
        """Test that empty token value is accepted (clears token)."""
        # Arrange
        from registry.main import app

        client = TestClient(app)

        # Mock service to return updated peer with cleared token
        updated_peer = sample_peer_config.model_copy()
        updated_peer.federation_token = None
        mock_peer_federation_service.get_peer_by_id.return_value = sample_peer_config
        mock_peer_federation_service.update_peer.return_value = updated_peer

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=mock_peer_federation_service,
        ):
            # Act - send empty string token
            response = client.patch(
                f"/api/peers/{sample_peer_config.peer_id}/token",
                json={"federation_token": ""},
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK

            # Verify service was called with empty string
            mock_peer_federation_service.update_peer.assert_called_once_with(
                sample_peer_config.peer_id,
                {"federation_token": ""},
            )

    @pytest.mark.asyncio
    async def test_update_peer_token_internal_error(
        self,
        mock_auth_admin,
        mock_peer_federation_service,
    ):
        """Test that internal errors return 400 with error message."""
        # Arrange
        from registry.main import app

        client = TestClient(app)

        # Mock service to raise generic ValueError during update
        mock_peer_federation_service.get_peer_by_id.return_value = None
        mock_peer_federation_service.update_peer.side_effect = ValueError("Internal database error")

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=mock_peer_federation_service,
        ):
            # Act
            response = client.patch(
                "/api/peers/test-peer/token",
                json={"federation_token": "new-token"},
            )

            # Assert
            assert response.status_code == status.HTTP_400_BAD_REQUEST


# =============================================================================
# Federation token is write-only on read/list responses (SSRF/leak hardening)
# =============================================================================


@pytest.mark.unit
class TestFederationTokenNeverSerialized:
    """The peer federation bearer token must never appear in a read response.

    GET /api/peers, GET /api/peers/{id}, and the create/update/enable/disable
    responses are callable by any principal with the federation/peers scope. They
    must return a redacted view (``has_federation_token`` boolean only) so a
    non-admin cannot harvest the plaintext token and impersonate this registry.
    """

    @pytest.mark.asyncio
    async def test_list_peers_omits_federation_token(
        self,
        mock_auth_admin,
        sample_peer_config,
    ):
        from registry.main import app

        client = TestClient(app)

        service = AsyncMock()
        service.list_peers.return_value = [sample_peer_config]

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=service,
        ):
            response = client.get("/api/peers")

        assert response.status_code == status.HTTP_200_OK
        body = response.text
        assert "original-token-abc123" not in body
        data = response.json()
        assert "federation_token" not in data[0]
        # Presence is surfaced as a non-sensitive boolean instead of the value.
        assert data[0]["has_federation_token"] is True
        assert data[0]["endpoint"] == "https://peer.example.com"

    @pytest.mark.asyncio
    async def test_get_peer_omits_federation_token(
        self,
        mock_auth_admin,
        sample_peer_config,
    ):
        from registry.main import app

        client = TestClient(app)

        service = AsyncMock()
        service.get_peer.return_value = sample_peer_config

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=service,
        ):
            response = client.get(f"/api/peers/{sample_peer_config.peer_id}")

        assert response.status_code == status.HTTP_200_OK
        assert "original-token-abc123" not in response.text
        data = response.json()
        assert "federation_token" not in data
        assert data["has_federation_token"] is True

    @pytest.mark.asyncio
    async def test_get_peer_without_token_reports_false(
        self,
        mock_auth_admin,
    ):
        from registry.main import app

        client = TestClient(app)

        no_token_peer = PeerRegistryConfig(
            peer_id="no-token-peer",
            name="No Token Peer",
            endpoint="https://peer2.example.com",
        )
        service = AsyncMock()
        service.get_peer.return_value = no_token_peer

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=service,
        ):
            response = client.get("/api/peers/no-token-peer")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "federation_token" not in data
        assert data["has_federation_token"] is False

    @pytest.mark.asyncio
    async def test_create_peer_response_omits_federation_token(
        self,
        mock_auth_admin,
        sample_peer_config,
    ):
        from registry.main import app

        client = TestClient(app)

        service = AsyncMock()
        service.add_peer.return_value = sample_peer_config

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=service,
        ):
            response = client.post(
                "/api/peers",
                json={
                    "peer_id": "test-peer",
                    "name": "Test Peer Registry",
                    "endpoint": "https://peer.example.com",
                    "federation_token": "original-token-abc123",
                },
            )

        assert response.status_code == status.HTTP_201_CREATED
        assert "original-token-abc123" not in response.text
        data = response.json()
        assert "federation_token" not in data
        assert data["has_federation_token"] is True

    @pytest.mark.asyncio
    async def test_update_peer_response_omits_federation_token(
        self,
        mock_auth_admin,
        sample_peer_config,
    ):
        from registry.main import app

        client = TestClient(app)

        service = AsyncMock()
        service.update_peer.return_value = sample_peer_config

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=service,
        ):
            response = client.put(
                "/api/peers/test-peer",
                json={"enabled": False},
            )

        assert response.status_code == status.HTTP_200_OK
        assert "original-token-abc123" not in response.text
        assert "federation_token" not in response.json()


# =============================================================================
# PUT /api/peers/{peer_id} log-redaction tests
# =============================================================================


@pytest.mark.unit
class TestUpdatePeerLogRedaction:
    """The peer update endpoint must never log a federation_token value."""

    _SECRET_TOKEN = "super-secret-federation-token-value-1234567890"

    @pytest.mark.asyncio
    async def test_update_peer_does_not_log_federation_token(
        self,
        mock_auth_admin,
        mock_peer_federation_service,
        sample_peer_config,
        caplog,
    ):
        """A PUT that carries a federation_token must not emit it to logs."""
        from registry.main import app

        client = TestClient(app)

        updated_peer = sample_peer_config.model_copy()
        mock_peer_federation_service.update_peer.return_value = updated_peer

        with patch(
            "registry.api.peer_management_routes.get_peer_federation_service",
            return_value=mock_peer_federation_service,
        ):
            with caplog.at_level(logging.DEBUG, logger="registry.api.peer_management_routes"):
                response = client.put(
                    f"/api/peers/{sample_peer_config.peer_id}",
                    json={
                        "enabled": True,
                        "federation_token": self._SECRET_TOKEN,
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        # The raw token must not appear anywhere in the captured logs, even at
        # DEBUG (the redacted payload is logged instead).
        combined = "\n".join(record.getMessage() for record in caplog.records)
        assert self._SECRET_TOKEN not in combined
        # The field NAME is still logged for diagnostics.
        assert "federation_token" in combined
        # The redaction marker appears in the DEBUG payload dump.
        assert "[REDACTED]" in combined
