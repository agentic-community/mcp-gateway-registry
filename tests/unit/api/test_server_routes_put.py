"""Route-level unit tests for PUT /servers/{path:path} (issue #1164).

Exercises ``update_server_endpoint`` via a FastAPI TestClient with the auth
dependency overridden. Service-layer collaborators (server_service,
registration_gate, security scanner, webhook) are mocked so these stay
unit-level.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api.server_routes import router

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def user_context() -> dict[str, Any]:
    """Owner of /test-server (matches sample_existing.registered_by)."""
    return {
        "username": "alice",
        "groups": ["g"],
        "scopes": ["modify_service"],
        "auth_method": "session",
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "ui_permissions": {
            "modify_service": ["all"],
            "publish_agent": ["all"],
        },
        "is_admin": False,
    }


@pytest.fixture
def admin_context() -> dict[str, Any]:
    return {
        "username": "admin",
        "groups": ["mcp-registry-admin"],
        "scopes": ["admin"],
        "auth_method": "session",
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "ui_permissions": {
            "modify_service": ["all"],
        },
        "is_admin": True,
    }


@pytest.fixture
def client(user_context):
    """TestClient with owner ``alice`` as the authenticated user."""
    app = FastAPI()
    app.include_router(router)

    from registry.api.server_routes import nginx_proxied_auth
    from registry.auth.csrf import verify_csrf_token_flexible

    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context
    app.dependency_overrides[verify_csrf_token_flexible] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_admin(admin_context):
    """TestClient with an admin user."""
    app = FastAPI()
    app.include_router(router)

    from registry.api.server_routes import nginx_proxied_auth
    from registry.auth.csrf import verify_csrf_token_flexible

    app.dependency_overrides[nginx_proxied_auth] = lambda: admin_context
    app.dependency_overrides[verify_csrf_token_flexible] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


def _existing(
    registered_by: str = "alice",
    updated_at: datetime | None = None,
    sync_metadata: dict[str, Any] | None = None,
    deployment: str = "remote",
    proxy_pass_url: str = "http://upstream:9000",
    auth_credential_encrypted: str | None = None,
) -> dict[str, Any]:
    """Build a stored server dict (as returned by server_service.get_server_info)."""
    ts = updated_at or datetime(2026, 1, 1, tzinfo=UTC)
    info: dict[str, Any] = {
        "server_name": "Existing Server",
        "description": "old description",
        "path": "/test-server",
        "tags": ["old"],
        "license": "MIT",
        "deployment": deployment,
        "registered_by": registered_by,
        "registered_at": ts.isoformat(),
        "updated_at": ts.isoformat(),
        "is_enabled": True,
        "version": "v1.0.0",
        "auth_scheme": "none",
    }
    if deployment == "remote":
        info["proxy_pass_url"] = proxy_pass_url
    if sync_metadata is not None:
        info["sync_metadata"] = sync_metadata
    if auth_credential_encrypted is not None:
        info["auth_credential_encrypted"] = auth_credential_encrypted
    return info


def _valid_put_body(**overrides: Any) -> dict[str, Any]:
    """Minimal valid PUT body."""
    body = {
        "server_name": "Existing Server",
        "description": "new description",
        "proxy_pass_url": "http://upstream:9000",
        "tags": ["new"],
        "license": "MIT",
        "visibility": "public",
    }
    body.update(overrides)
    return body


def _gate_allow() -> MagicMock:
    return MagicMock(allowed=True, error_message=None)


def _gate_deny(message: str = "blocked by policy") -> MagicMock:
    return MagicMock(allowed=False, error_message=message)


def _patch_gate_and_webhook():
    """Patch the registration gate and webhook so they don't run real I/O."""
    return [
        patch(
            "registry.api.server_routes.check_registration_gate",
            AsyncMock(return_value=_gate_allow()),
        ),
        patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
    ]


# =============================================================================
# Happy path + audit
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestServerPut:
    """Tests for PUT /servers/{path:path}."""

    def test_put_full_replacement_happy_path(self, client):
        existing = _existing()
        fresh = {**existing, "description": "new description"}

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, fresh])
            svc.update_server = AsyncMock(return_value=True)

            response = client.put("/servers/test-server", json=_valid_put_body())

        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "new description"
        assert data["server_name"] == "Existing Server"

    def test_put_audit_action_set_with_existing_server_name(self, client):
        existing = _existing()
        existing["server_name"] = "Original Name"

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
            patch("registry.api.server_routes.set_audit_action") as mock_audit,
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            client.put(
                "/servers/test-server",
                json=_valid_put_body(server_name="Brand New Name"),
            )

        mock_audit.assert_called_once()
        # description includes EXISTING server_name, not new one
        assert "Original Name" in mock_audit.call_args.kwargs["description"]
        assert "Brand New Name" not in mock_audit.call_args.kwargs["description"]

    def test_put_audit_metadata_had_if_match_false_when_absent(self, client):
        existing = _existing()
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
            patch("registry.api.server_routes.set_audit_action") as mock_audit,
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            client.put("/servers/test-server", json=_valid_put_body())

        assert mock_audit.call_args.kwargs["metadata"] == {"had_if_match": False}

    def test_put_audit_metadata_had_if_match_true_when_present(self, client):
        existing = _existing()
        # Use the matching ETag so we don't 412 first.
        ts_ms = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
        if_match = f'W/"{ts_ms}"'

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
            patch("registry.api.server_routes.set_audit_action") as mock_audit,
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            client.put(
                "/servers/test-server",
                json=_valid_put_body(),
                headers={"If-Match": if_match},
            )

        assert mock_audit.call_args.kwargs["metadata"] == {"had_if_match": True}

    def test_put_404_when_server_not_found(self, client):
        with patch("registry.api.server_routes.server_service") as svc:
            svc.get_server_info = AsyncMock(return_value=None)
            response = client.put("/servers/nope", json=_valid_put_body())

        assert response.status_code == 404
        assert "/nope" in response.json()["detail"]

    def test_put_403_when_not_owner_and_not_admin(self, client):
        existing = _existing(registered_by="bob")
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.put("/servers/test-server", json=_valid_put_body())

        assert response.status_code == 403
        assert "only update servers you registered" in response.json()["detail"]

    def test_put_403_when_missing_modify_service_permission(self, client):
        existing = _existing()
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.api.server_routes.user_has_asset_permission",
                return_value=False,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.put("/servers/test-server", json=_valid_put_body())

        assert response.status_code == 403
        assert "permission" in response.json()["detail"].lower()

    def test_put_admin_can_update_any_server(self, client_admin):
        existing = _existing(registered_by="bob")  # admin is not the owner
        fresh = {**existing, "description": "new description"}

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, fresh])
            svc.update_server = AsyncMock(return_value=True)

            response = client_admin.put("/servers/test-server", json=_valid_put_body())

        assert response.status_code == 200

    def test_put_federated_guard_blocks(self, client):
        existing = _existing(
            sync_metadata={"is_federated": True, "source_peer_id": "peer-99"},
        )
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.put("/servers/test-server", json=_valid_put_body())

        assert response.status_code == 403
        assert "peer-99" in response.json()["detail"]

    def test_put_422_extra_field(self, client):
        existing = _existing()
        body = _valid_put_body()
        body["totally_unknown_field"] = "boom"

        with patch("registry.api.server_routes.server_service") as svc:
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.put("/servers/test-server", json=body)

        assert response.status_code == 422

    def test_put_422_credential_fields_rejected(self, client):
        existing = _existing()
        body = _valid_put_body()
        body["auth_credential"] = "Bearer leaked-secret"

        with patch("registry.api.server_routes.server_service") as svc:
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.put("/servers/test-server", json=body)

        assert response.status_code == 422

    def test_put_422_registrant_only_fields_rejected(self, client):
        existing = _existing()
        body = _valid_put_body()
        body["registered_by"] = "not-allowed"

        with patch("registry.api.server_routes.server_service") as svc:
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.put("/servers/test-server", json=body)

        assert response.status_code == 422

    def test_put_412_if_match_mismatch(self, client):
        existing = _existing(updated_at=datetime(2026, 1, 1, tzinfo=UTC))
        stale_ms = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000)
        stale_etag = f'W/"{stale_ms}"'

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.put(
                "/servers/test-server",
                json=_valid_put_body(),
                headers={"If-Match": stale_etag},
            )

        assert response.status_code == 412
        assert "If-Match" in response.json()["detail"]

    def test_put_400_if_match_malformed(self, client):
        existing = _existing()
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.put(
                "/servers/test-server",
                json=_valid_put_body(),
                headers={"If-Match": "not-an-etag"},
            )

        assert response.status_code == 400
        assert "Malformed If-Match" in response.json()["detail"]

    def test_put_400_if_match_strong_form(self, client):
        existing = _existing()
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.put(
                "/servers/test-server",
                json=_valid_put_body(),
                headers={"If-Match": '"1234567890"'},
            )

        assert response.status_code == 400
        assert "Strong ETag" in response.json()["detail"]

    def test_put_etag_header_set_on_response(self, client):
        ts = datetime(2026, 5, 5, tzinfo=UTC)
        existing = _existing(updated_at=ts)
        # Fresh has a newer updated_at the handler will read.
        fresh = {**existing, "updated_at": datetime(2026, 6, 6, tzinfo=UTC).isoformat()}

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, fresh])
            svc.update_server = AsyncMock(return_value=True)

            response = client.put("/servers/test-server", json=_valid_put_body())

        assert response.status_code == 200
        etag = response.headers.get("ETag")
        assert etag is not None
        assert etag.startswith('W/"')
        # The ETag value is digits only inside the quotes.
        inner = etag[3:-1]
        assert inner.isdigit()

    def test_put_credentials_preserved_when_not_supplied(self, client):
        """Stored ``auth_credential_encrypted`` survives a PUT without that
        field in the body. Defence-in-depth: PUT body has no credential
        fields, but the stored encrypted blob must still be present in the
        merged dict the service receives."""
        existing = _existing(auth_credential_encrypted="ENC::abc123")
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            response = client.put("/servers/test-server", json=_valid_put_body())

        assert response.status_code == 200
        merged = svc.update_server.call_args[0][1]
        assert merged["auth_credential_encrypted"] == "ENC::abc123"

    def test_put_calls_server_service_update_server(self, client):
        existing = _existing()
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            client.put("/servers/test-server", json=_valid_put_body())

        svc.update_server.assert_called_once()
        called_path, called_dict = svc.update_server.call_args[0]
        assert called_path == "/test-server"
        assert isinstance(called_dict, dict)
        assert called_dict["description"] == "new description"

    def test_put_returns_500_when_update_fails(self, client):
        existing = _existing()
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            svc.update_server = AsyncMock(return_value=False)

            response = client.put("/servers/test-server", json=_valid_put_body())

        assert response.status_code == 500
        assert "Failed to save" in response.json()["detail"]

    def test_put_registration_gate_denial_403(self, client):
        existing = _existing()
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_deny("policy says no")),
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            svc.update_server = AsyncMock(return_value=True)

            response = client.put("/servers/test-server", json=_valid_put_body())

        assert response.status_code == 403
        assert "policy says no" in response.json()["detail"]

    def test_put_security_scan_triggered_when_proxy_url_changes(self, client):
        existing = _existing(proxy_pass_url="http://old:9000")
        # Body has a different proxy_pass_url — should trigger scan.
        body = _valid_put_body(proxy_pass_url="http://new:9000")

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
            patch(
                "registry.api.server_routes._perform_security_scan_on_registration",
                new_callable=AsyncMock,
            ) as mock_scan,
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            response = client.put("/servers/test-server", json=body)

        assert response.status_code == 200
        # The scan coroutine was scheduled (asyncio.create_task wraps it).
        mock_scan.assert_called_once()
        scan_args = mock_scan.call_args[0]
        assert scan_args[0] == "/test-server"
        assert scan_args[1] == "http://new:9000"

    def test_put_security_scan_not_triggered_for_local_deployment(self, client):
        """For local deployment, the scan must NOT run regardless of body
        contents. proxy_pass_url is ignored for local servers."""
        existing = _existing(deployment="local", proxy_pass_url="http://anything")
        # Even if body carries a proxy URL, local existing skips scan.
        body = _valid_put_body(proxy_pass_url="http://different:9000")

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
            patch(
                "registry.api.server_routes._perform_security_scan_on_registration",
                new_callable=AsyncMock,
            ) as mock_scan,
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            response = client.put("/servers/test-server", json=body)

        assert response.status_code == 200
        mock_scan.assert_not_called()

    def test_put_webhook_scheduled_on_success(self, client):
        existing = _existing(auth_credential_encrypted="ENC::secret")
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch(
                "registry.api.server_routes.send_registration_webhook",
                new_callable=AsyncMock,
            ) as mock_webhook,
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            response = client.put("/servers/test-server", json=_valid_put_body())

        assert response.status_code == 200
        mock_webhook.assert_called_once()
        kwargs = mock_webhook.call_args.kwargs
        assert kwargs["event_type"] == "update"
        assert kwargs["registration_type"] == "server"
        assert kwargs["performed_by"] == "alice"
        # Credentials are stripped from the webhook payload.
        card_data = kwargs["card_data"]
        assert "auth_credential_encrypted" not in card_data
